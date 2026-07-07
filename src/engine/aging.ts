// Face re-aging via FRAN (U-Net that predicts an aging *delta* conditioned on source and
// target age). Delta-based editing preserves identity and native detail: the model runs on a
// 1024px face crop in 512px sliding windows, and its upsampled delta is added onto the
// ORIGINAL native pixels under a soft face mask — nothing is regenerated.
// Weights: MIT-licensed reimplementation (timroelofs123/face_re-aging), CI-converted to ONNX.
import * as ort from 'onnxruntime-web'
import { MODELS } from './models'
import { createOrtSession } from './ort'
import { progressChannel } from './util'
import { makeCanvas, rasterizeOval, insideFeather } from './mask'
import type { Face } from './types'

const CROP = 1024
const WIN = 512
const STRIDE = 256

const channel = progressChannel()
export const onAgingProgress = channel.on

let sessionPromise: Promise<ort.InferenceSession> | null = null

function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      try {
        channel.emit({ loading: true, frac: 0 })
        const s = await createOrtSession(MODELS.fran, (f) =>
          channel.emit({ loading: true, frac: f }),
        )
        channel.emit({ loading: false, frac: 1 })
        return s
      } catch (e) {
        channel.emit({ loading: false, frac: 1 })
        sessionPromise = null
        throw e
      }
    })()
  }
  return sessionPromise
}

/** The model ships only after the one-time conversion workflow has run — probe for it. */
export async function isAgingAvailable(): Promise<boolean> {
  try {
    const res = await fetch(MODELS.fran, { headers: { Range: 'bytes=0-0' } })
    return res.ok || res.status === 206
  } catch {
    return false
  }
}

/**
 * Age the face and return native-resolution pixels (RGBA) for the whole image.
 * `base` lets callers age an already-edited frame; defaults to the face bitmap.
 */
export async function computeAge(
  face: Face,
  sourceAge: number,
  targetAge: number,
  onProgress?: (frac: number) => void,
): Promise<ImageData> {
  if (!face.landmarks) throw new Error('No landmarks for the selected face')
  const session = await getSession()
  const W = face.width
  const H = face.height

  // Face crop with generous margins (network was trained on loosely-framed squares).
  const b = face.landmarks.box
  let x0 = b.x - b.width * 0.35
  let x1 = b.x + b.width * 1.35
  let y0 = b.y - b.height * 0.55
  let y1 = b.y + b.height * 1.35
  // Square it (expand the shorter side around its center).
  const cw = x1 - x0
  const ch = y1 - y0
  if (cw > ch) {
    const pad = (cw - ch) / 2
    y0 -= pad
    y1 += pad
  } else {
    const pad = (ch - cw) / 2
    x0 -= pad
    x1 += pad
  }
  x0 = Math.max(0, Math.round(x0))
  y0 = Math.max(0, Math.round(y0))
  x1 = Math.min(W, Math.round(x1))
  y1 = Math.min(H, Math.round(y1))
  const cwN = x1 - x0
  const chN = y1 - y0
  if (cwN < 32 || chN < 32) throw new Error('Face too small to age')

  // Resample the crop to 1024 for the model.
  const cropCanvas = makeCanvas(CROP, CROP)
  const cctx = cropCanvas.getContext('2d') as CanvasRenderingContext2D
  cctx.drawImage(face.bitmap as CanvasImageSource, x0, y0, cwN, chN, 0, 0, CROP, CROP)
  const cropPx = cctx.getImageData(0, 0, CROP, CROP).data

  const src01 = Math.min(1, Math.max(0, sourceAge / 100))
  const tgt01 = Math.min(1, Math.max(0, targetAge / 100))

  // Sliding 512 windows at stride 256 over the 1024 crop; average overlapping deltas.
  const delta = new Float32Array(CROP * CROP * 3)
  const count = new Float32Array(CROP * CROP)
  const steps: number[] = []
  for (let s = 0; s + WIN <= CROP; s += STRIDE) steps.push(s)
  const total = steps.length * steps.length
  let done = 0

  const plane = WIN * WIN
  for (const wy of steps) {
    for (const wx of steps) {
      const input = new Float32Array(5 * plane)
      for (let y = 0; y < WIN; y++) {
        const row = (wy + y) * CROP
        for (let x = 0; x < WIN; x++) {
          const p = y * WIN + x
          const o = (row + wx + x) * 4
          input[p] = cropPx[o] / 255
          input[plane + p] = cropPx[o + 1] / 255
          input[2 * plane + p] = cropPx[o + 2] / 255
        }
      }
      input.fill(src01, 3 * plane, 4 * plane)
      input.fill(tgt01, 4 * plane, 5 * plane)

      const tensor = new ort.Tensor('float32', input, [1, 5, WIN, WIN])
      const outputs = await session.run({ [session.inputNames[0]]: tensor })
      const outD = outputs[session.outputNames[0]].data as Float32Array

      for (let y = 0; y < WIN; y++) {
        const row = (wy + y) * CROP
        for (let x = 0; x < WIN; x++) {
          const p = y * WIN + x
          const q = row + wx + x
          delta[q * 3] += outD[p]
          delta[q * 3 + 1] += outD[plane + p]
          delta[q * 3 + 2] += outD[2 * plane + p]
          count[q] += 1
        }
      }
      done++
      onProgress?.(done / total)
      await new Promise((r) => setTimeout(r, 0)) // keep the UI responsive
    }
  }
  for (let q = 0; q < count.length; q++) {
    const c = Math.max(1, count[q])
    delta[q * 3] /= c
    delta[q * 3 + 1] /= c
    delta[q * 3 + 2] /= c
  }

  // Soft face mask at native res (oval grown to include hair/neck, feathered inward).
  const oval = rasterizeOval(face.landmarks.points, W, H, 1.35)
  const m = insideFeather(oval, W, H, Math.max(2, Math.round(b.width * 0.08)))

  // Compose: native pixels + bilinearly-upsampled delta, only inside the mask & crop.
  const outCanvas = makeCanvas(W, H)
  const octx = outCanvas.getContext('2d') as CanvasRenderingContext2D
  octx.drawImage(face.bitmap as CanvasImageSource, 0, 0)
  const out = octx.getImageData(0, 0, W, H)
  const d = out.data
  const sx = CROP / cwN
  const sy = CROP / chN
  for (let y = y0; y < y1; y++) {
    const fy = Math.min(CROP - 1.001, (y - y0 + 0.5) * sy - 0.5)
    const iy = Math.floor(fy)
    const ty = fy - iy
    for (let x = x0; x < x1; x++) {
      const j = y * W + x
      const mv = m[j]
      if (mv <= 0) continue
      const fx = Math.min(CROP - 1.001, (x - x0 + 0.5) * sx - 0.5)
      const ix = Math.floor(fx)
      const tx = fx - ix
      const o = j * 4
      for (let c = 0; c < 3; c++) {
        const d00 = delta[(iy * CROP + ix) * 3 + c]
        const d10 = delta[(iy * CROP + ix + 1) * 3 + c]
        const d01 = delta[((iy + 1) * CROP + ix) * 3 + c]
        const d11 = delta[((iy + 1) * CROP + ix + 1) * 3 + c]
        const dv = d00 * (1 - tx) * (1 - ty) + d10 * tx * (1 - ty) + d01 * (1 - tx) * ty + d11 * tx * ty
        const v = d[o + c] + dv * 255 * mv
        d[o + c] = v < 0 ? 0 : v > 255 ? 255 : v
      }
    }
  }
  return out
}
