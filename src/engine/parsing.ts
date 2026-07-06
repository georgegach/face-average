// BiSeNet face parsing (19 CelebAMask-HQ classes at 512x512) via onnxruntime-web.
// Powers the Edit mode's mask-driven tools: masks are computed once per face at 512 and
// resampled to native resolution, so the pixel edits themselves stay full-quality.
import * as ort from 'onnxruntime-web'
import { MODELS } from './models'
import { fetchWithProgressCached } from './download'
import { makeCanvas, blurMask } from './mask'
import type { Face } from './types'

export const PARSE_SIZE = 512

// CelebAMask-HQ label indices (face-parsing.PyTorch convention).
export const CLS = {
  background: 0,
  skin: 1,
  lBrow: 2,
  rBrow: 3,
  lEye: 4,
  rEye: 5,
  glasses: 6,
  lEar: 7,
  rEar: 8,
  earring: 9,
  nose: 10,
  mouth: 11, // inner mouth (teeth)
  uLip: 12,
  lLip: 13,
  neck: 14,
  necklace: 15,
  cloth: 16,
  hair: 17,
  hat: 18,
} as const

export interface Parsing {
  labels: Uint8Array // PARSE_SIZE * PARSE_SIZE argmax label map
  scale: number // native px -> parse px scale factor
  dx: number // letterbox x offset in parse space
  dy: number // letterbox y offset in parse space
}

// Load-progress broadcast for the one-time model download (mirrors landmarks.ts).
export type ParseLoadState = { loading: boolean; frac: number }
const listeners = new Set<(s: ParseLoadState) => void>()
export function onParsingProgress(cb: (s: ParseLoadState) => void): () => void {
  listeners.add(cb)
  return () => listeners.delete(cb)
}
function emit(s: ParseLoadState) {
  for (const cb of listeners) cb(s)
}

ort.env.wasm.wasmPaths = MODELS.ortWasm

let sessionPromise: Promise<ort.InferenceSession> | null = null

function providers(): string[] {
  const p: string[] = []
  if ('gpu' in navigator) p.push('webgpu')
  p.push('wasm')
  return p
}

function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    ort.env.wasm.numThreads = 1 // GitHub Pages has no cross-origin isolation
    sessionPromise = (async () => {
      try {
        emit({ loading: true, frac: 0 })
        const bytes = await fetchWithProgressCached(MODELS.faceParsing, (f) =>
          emit({ loading: true, frac: f }),
        )
        const s = await ort.InferenceSession.create(bytes, {
          executionProviders: providers(),
        })
        emit({ loading: false, frac: 1 })
        return s
      } catch (e) {
        emit({ loading: false, frac: 1 })
        sessionPromise = null // allow retry after a transient failure
        throw e
      }
    })()
  }
  return sessionPromise
}

// ImageNet normalization used by face-parsing.PyTorch / yakhyo's export.
const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]

async function runParse(bitmap: ImageBitmap): Promise<Parsing> {
  const session = await getSession()

  // Letterbox the image into a 512x512 square (preserves aspect; black padding).
  const scale = PARSE_SIZE / Math.max(bitmap.width, bitmap.height)
  const w = Math.max(1, Math.round(bitmap.width * scale))
  const h = Math.max(1, Math.round(bitmap.height * scale))
  const dx = Math.floor((PARSE_SIZE - w) / 2)
  const dy = Math.floor((PARSE_SIZE - h) / 2)

  const canvas = makeCanvas(PARSE_SIZE, PARSE_SIZE)
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, PARSE_SIZE, PARSE_SIZE)
  ctx.drawImage(bitmap as CanvasImageSource, dx, dy, w, h)
  const px = ctx.getImageData(0, 0, PARSE_SIZE, PARSE_SIZE).data

  const plane = PARSE_SIZE * PARSE_SIZE
  const input = new Float32Array(3 * plane)
  for (let p = 0; p < plane; p++) {
    const o = p * 4
    input[p] = (px[o] / 255 - MEAN[0]) / STD[0]
    input[plane + p] = (px[o + 1] / 255 - MEAN[1]) / STD[1]
    input[2 * plane + p] = (px[o + 2] / 255 - MEAN[2]) / STD[2]
  }

  const tensor = new ort.Tensor('float32', input, [1, 3, PARSE_SIZE, PARSE_SIZE])
  const outputs = await session.run({ [session.inputNames[0]]: tensor })
  const out = outputs[session.outputNames[0]]
  const dims = out.dims as number[]
  const nCls = dims[1]
  const data = out.data as Float32Array

  // Argmax over the class dimension -> label map.
  const labels = new Uint8Array(plane)
  for (let p = 0; p < plane; p++) {
    let best = 0
    let bestV = data[p]
    for (let c = 1; c < nCls; c++) {
      const v = data[c * plane + p]
      if (v > bestV) {
        bestV = v
        best = c
      }
    }
    labels[p] = best
  }

  return { labels, scale, dx, dy }
}

// Parsing is deterministic per bitmap, so cache by face id (bitmaps never change per id).
const cache = new Map<string, Promise<Parsing>>()

export function getParsing(face: Face): Promise<Parsing> {
  let p = cache.get(face.id)
  if (!p) {
    p = runParse(face.bitmap).catch((e) => {
      cache.delete(face.id) // don't poison the cache on transient failure
      throw e
    })
    cache.set(face.id, p)
  }
  return p
}

/**
 * Build a soft native-resolution (W x H) 0..1 mask covering the given classes,
 * feathered by `featherPx`. Nearest-neighbour sampling from the 512 label map.
 */
export function classMask(
  p: Parsing,
  W: number,
  H: number,
  classes: readonly number[],
  featherPx: number,
): Float32Array {
  const inCls = new Uint8Array(32)
  for (const c of classes) inCls[c] = 1
  const m = new Float32Array(W * H)
  for (let y = 0; y < H; y++) {
    const py = Math.min(PARSE_SIZE - 1, Math.max(0, Math.round(y * p.scale) + p.dy))
    const row = py * PARSE_SIZE
    for (let x = 0; x < W; x++) {
      const pxi = Math.min(PARSE_SIZE - 1, Math.max(0, Math.round(x * p.scale) + p.dx))
      m[y * W + x] = inCls[p.labels[row + pxi]]
    }
  }
  return featherPx >= 1 ? blurMask(m, W, H, Math.round(featherPx)) : m
}
