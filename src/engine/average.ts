import Delaunator from 'delaunator'
import {
  boundaryPoints,
  canonicalEyes,
  irisCenters,
  similarityFrom2,
  transformLandmarks,
  N_TOTAL,
  type Pt,
} from './geometry'
import { getWarpEngine } from './warp'
import { computeStats, averageStats, applyTransfer } from './color'
import { N_LANDMARKS, type AverageSettings, type Face } from './types'

export interface AverageResult {
  imageData: ImageData
  meshUsed: Float32Array // destination mesh (N_TOTAL*2) for the editor/overlay
  count: number
}

interface Prepared {
  face: Face
  aligned: OffscreenCanvas | HTMLCanvasElement
  points: Float32Array // N_TOTAL*2 in output space
  weight: number
}

function makeCanvas(w: number, h: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h)
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  return c
}

function prepare(face: Face, s: AverageSettings, dstEyes: { right: Pt; left: Pt }): Prepared | null {
  if (!face.landmarks) return null
  const { right, left } = irisCenters(face.landmarks)
  const m = similarityFrom2(right, left, dstEyes.right, dstEyes.left)

  const canvas = makeCanvas(s.outWidth, s.outHeight)
  const ctx = canvas.getContext('2d') as
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D
  ctx.clearRect(0, 0, s.outWidth, s.outHeight)
  ctx.setTransform(m.a, m.c, m.b, m.d, m.tx, m.ty)
  ctx.drawImage(face.bitmap as CanvasImageSource, 0, 0)
  ctx.setTransform(1, 0, 0, 1, 0, 0)

  const lm = transformLandmarks(face.landmarks, m)
  const bnd = boundaryPoints(s.outWidth, s.outHeight)
  const points = new Float32Array(N_TOTAL * 2)
  points.set(lm, 0)
  points.set(bnd, N_LANDMARKS * 2)

  return { face, aligned: canvas, points, weight: Math.max(0, face.weight) }
}

export function computeAverage(faces: Face[], s: AverageSettings): AverageResult {
  const usable = faces.filter((f) => f.enabled && f.landmarks && !f.failed)
  if (usable.length === 0) throw new Error('No faces with detected landmarks')

  const dstEyes = canonicalEyes(s.outWidth, s.outHeight, s.eyeDistance, s.eyeRatioY)
  const prepared = usable.map((f) => prepare(f, s, dstEyes)).filter((p): p is Prepared => !!p)
  if (prepared.length === 0) throw new Error('Alignment failed')

  const w = s.outWidth
  const h = s.outHeight
  const totalWeight = prepared.reduce((a, p) => a + p.weight, 0) || 1

  // Destination mesh: template geometry, or the weighted average of all meshes.
  const mesh = new Float32Array(N_TOTAL * 2)
  const template = s.templateId ? prepared.find((p) => p.face.id === s.templateId) : null
  if (template) {
    mesh.set(template.points)
  } else {
    for (const p of prepared) {
      const wgt = p.weight / totalWeight
      for (let i = 0; i < N_TOTAL * 2; i++) mesh[i] += p.points[i] * wgt
    }
  }
  // Boundary points stay exactly on the canvas edge regardless of weighting.
  const bnd = boundaryPoints(w, h)
  mesh.set(bnd, N_LANDMARKS * 2)

  // Shared triangulation over the destination mesh.
  const coords = new Float64Array(N_TOTAL * 2)
  for (let i = 0; i < N_TOTAL * 2; i++) coords[i] = mesh[i]
  const del = new Delaunator(coords)
  const tris = new Uint32Array(del.triangles)

  // Pass 1: warp every face, gather stats for colour normalisation.
  const engine = getWarpEngine()
  const warped: Uint8ClampedArray[] = []
  const stats = []
  for (const p of prepared) {
    const px = engine.warp(p.aligned, p.points, mesh, tris, w, h)
    warped.push(px)
    if (s.colorNormalize) stats.push(computeStats(px))
  }
  const target = s.colorNormalize ? averageStats(stats) : null

  // Pass 2: accumulate weighted colour + coverage.
  const acc = new Float32Array(w * h * 3)
  const wacc = new Float32Array(w * h)
  for (let k = 0; k < prepared.length; k++) {
    const px = warped[k]
    if (target && stats[k].count > 0) applyTransfer(px, stats[k], target, 0.7)
    const weight = prepared[k].weight
    for (let i = 0, j = 0; i < px.length; i += 4, j++) {
      const a = px[i + 3] / 255
      if (a === 0) continue
      const cw = a * weight
      acc[j * 3] += px[i] * cw
      acc[j * 3 + 1] += px[i + 1] * cw
      acc[j * 3 + 2] += px[i + 2] * cw
      wacc[j] += cw
    }
  }

  const out = new ImageData(w, h)
  const od = out.data
  for (let j = 0; j < wacc.length; j++) {
    const wv = wacc[j]
    const o = j * 4
    if (wv > 1e-4) {
      od[o] = acc[j * 3] / wv
      od[o + 1] = acc[j * 3 + 1] / wv
      od[o + 2] = acc[j * 3 + 2] / wv
      od[o + 3] = 255
    } else {
      od[o + 3] = 0 // uncovered — filled by background pass
    }
  }

  fillBackground(out, s)
  return { imageData: out, meshUsed: mesh, count: prepared.length }
}

function fillBackground(img: ImageData, s: AverageSettings) {
  if (s.background === 'transparent') return
  const { width: w, height: h, data } = img
  if (s.background === 'studio') {
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] === 0) {
        data[i] = 18
        data[i + 1] = 21
        data[i + 2] = 26
        data[i + 3] = 255
      }
    }
    return
  }
  // 'blur': fill holes with a heavily box-blurred copy of the covered pixels.
  const blurred = boxBlur(data, w, h, 24)
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) {
      data[i] = blurred[i]
      data[i + 1] = blurred[i + 1]
      data[i + 2] = blurred[i + 2]
      data[i + 3] = 255
    }
  }
}

function boxBlur(src: Uint8ClampedArray, w: number, h: number, r: number): Uint8ClampedArray {
  // Cheap separable box blur that ignores transparent source pixels.
  const tmp = new Float32Array(w * h * 3)
  const tw = new Float32Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x
      let r0 = 0,
        g0 = 0,
        b0 = 0,
        wsum = 0
      for (let dx = -r; dx <= r; dx += 4) {
        const xx = x + dx
        if (xx < 0 || xx >= w) continue
        const si = (y * w + xx) * 4
        if (src[si + 3] === 0) continue
        r0 += src[si]
        g0 += src[si + 1]
        b0 += src[si + 2]
        wsum++
      }
      tmp[idx * 3] = r0
      tmp[idx * 3 + 1] = g0
      tmp[idx * 3 + 2] = b0
      tw[idx] = wsum
    }
  }
  const out = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x
      let r0 = 0,
        g0 = 0,
        b0 = 0,
        wsum = 0
      for (let dy = -r; dy <= r; dy += 4) {
        const yy = y + dy
        if (yy < 0 || yy >= h) continue
        const si = yy * w + x
        if (tw[si] === 0) continue
        r0 += tmp[si * 3]
        g0 += tmp[si * 3 + 1]
        b0 += tmp[si * 3 + 2]
        wsum += tw[si]
      }
      const o = idx * 4
      if (wsum > 0) {
        out[o] = r0 / wsum
        out[o + 1] = g0 / wsum
        out[o + 2] = b0 / wsum
        out[o + 3] = 255
      }
    }
  }
  return out
}
