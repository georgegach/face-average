// Lazy ONNX upscaler (Real-ESRGAN family). Tiled to bound memory; prefers the
// WebGPU execution provider when available, falling back to wasm+SIMD.
// Models are optional assets — if absent, callers get a clear error and the
// feature stays disabled rather than breaking the app.
import * as ort from 'onnxruntime-web'
import { MODELS, type UpscalerKind } from './models'
import { fetchWithProgress } from './download'

const SCALE = 4
const TILE = 192
const OVERLAP = 16

// Load ORT's own wasm/mjs assets same-origin (self-hosted), not from a CDN.
ort.env.wasm.wasmPaths = MODELS.ortWasm

const sessions = new Map<UpscalerKind, Promise<ort.InferenceSession>>()

function providers(): string[] {
  const p: string[] = []
  if ('gpu' in navigator) p.push('webgpu')
  p.push('wasm')
  return p
}

async function getSession(
  kind: UpscalerKind,
  onDownload?: (frac: number) => void,
): Promise<ort.InferenceSession> {
  let s = sessions.get(kind)
  if (!s) {
    ort.env.wasm.numThreads = 1 // GitHub Pages has no cross-origin isolation
    s = (async () => {
      const bytes = await fetchWithProgress(MODELS.upscalers[kind], onDownload)
      return ort.InferenceSession.create(bytes, { executionProviders: providers() })
    })()
    sessions.set(kind, s)
  } else {
    onDownload?.(1) // already downloaded/cached
  }
  return s
}

export async function isUpscalerAvailable(kind: UpscalerKind): Promise<boolean> {
  // Ranged GET (1 byte) — HEAD isn't reliable through HuggingFace's redirect.
  try {
    const res = await fetch(MODELS.upscalers[kind], { headers: { Range: 'bytes=0-0' } })
    return res.ok || res.status === 206
  } catch {
    return false
  }
}

function tileToTensor(data: Uint8ClampedArray, w: number, h: number): ort.Tensor {
  const arr = new Float32Array(3 * w * h)
  const plane = w * h
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    arr[p] = data[i] / 255
    arr[plane + p] = data[i + 1] / 255
    arr[2 * plane + p] = data[i + 2] / 255
  }
  return new ort.Tensor('float32', arr, [1, 3, h, w])
}

function tensorToTile(t: ort.Tensor, out: Uint8ClampedArray, ow: number) {
  const [, , h, w] = t.dims as number[]
  const plane = w * h
  const d = t.data as Float32Array
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x
      const o = (y * ow + x) * 4
      out[o] = clamp255(d[p] * 255)
      out[o + 1] = clamp255(d[plane + p] * 255)
      out[o + 2] = clamp255(d[2 * plane + p] * 255)
      out[o + 3] = 255
    }
  }
}

const clamp255 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v)

export type UpscaleStage = 'download' | 'upscale'

export async function upscale(
  img: ImageData,
  kind: UpscalerKind,
  onProgress?: (stage: UpscaleStage, frac: number) => void,
): Promise<ImageData> {
  const session = await getSession(kind, (f) => onProgress?.('download', f))
  const inName = session.inputNames[0]
  const outName = session.outputNames[0]
  const { width: W, height: H } = img
  const outW = W * SCALE
  const outH = H * SCALE
  const result = new ImageData(outW, outH)

  const step = TILE - OVERLAP
  const cols = Math.ceil(W / step)
  const rows = Math.ceil(H / step)
  const total = cols * rows
  let done = 0

  const src = img.data
  for (let ty = 0; ty < rows; ty++) {
    for (let tx = 0; tx < cols; tx++) {
      const sx = Math.min(tx * step, Math.max(0, W - TILE))
      const sy = Math.min(ty * step, Math.max(0, H - TILE))
      const tw = Math.min(TILE, W - sx)
      const th = Math.min(TILE, H - sy)

      const tile = new Uint8ClampedArray(tw * th * 4)
      for (let y = 0; y < th; y++) {
        const srcOff = ((sy + y) * W + sx) * 4
        tile.set(src.subarray(srcOff, srcOff + tw * 4), y * tw * 4)
      }

      const input = tileToTensor(tile, tw, th)
      const outputs = await session.run({ [inName]: input })
      const outT = outputs[outName]
      const [, , oh, ow] = outT.dims as number[]
      const outTile = new Uint8ClampedArray(ow * oh * 4)
      tensorToTile(outT, outTile, ow)

      // Blit tile (crop overlap on inner edges to avoid double seams).
      const cropL = tx > 0 ? (OVERLAP / 2) * SCALE : 0
      const cropT = ty > 0 ? (OVERLAP / 2) * SCALE : 0
      const dx = sx * SCALE + cropL
      const dy = sy * SCALE + cropT
      for (let y = cropT; y < oh; y++) {
        for (let x = cropL; x < ow; x++) {
          const dX = dx + (x - cropL)
          const dY = dy + (y - cropT)
          if (dX >= outW || dY >= outH) continue
          const o = (dY * outW + dX) * 4
          const s = (y * ow + x) * 4
          result.data[o] = outTile[s]
          result.data[o + 1] = outTile[s + 1]
          result.data[o + 2] = outTile[s + 2]
          result.data[o + 3] = 255
        }
      }
      done++
      onProgress?.('upscale', done / total)
      await new Promise((r) => setTimeout(r, 0)) // yield to keep UI responsive
    }
  }
  return result
}
