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
import { N_LANDMARKS, type AverageSettings, type Face } from './types'

interface Side {
  aligned: OffscreenCanvas | HTMLCanvasElement
  points: Float32Array
}

function alignSide(face: Face, s: AverageSettings, dstEyes: { right: Pt; left: Pt }): Side {
  if (!face.landmarks) throw new Error('face missing landmarks')
  const { right, left } = irisCenters(face.landmarks)
  const m = similarityFrom2(right, left, dstEyes.right, dstEyes.left)
  const canvas =
    typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(s.outWidth, s.outHeight)
      : Object.assign(document.createElement('canvas'), { width: s.outWidth, height: s.outHeight })
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D
  ctx.setTransform(m.a, m.c, m.b, m.d, m.tx, m.ty)
  ctx.drawImage(face.bitmap as CanvasImageSource, 0, 0)
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  const lm = transformLandmarks(face.landmarks, m)
  const points = new Float32Array(N_TOTAL * 2)
  points.set(lm, 0)
  points.set(boundaryPoints(s.outWidth, s.outHeight), N_LANDMARKS * 2)
  return { aligned: canvas, points }
}

/** Prepared two-face morph; renderFrame(t) is real-time for slider scrubbing. */
export class MorphSession {
  private a: Side
  private b: Side
  private tris: Uint32Array
  private w: number
  private h: number

  constructor(faceA: Face, faceB: Face, s: AverageSettings) {
    const dstEyes = canonicalEyes(s.outWidth, s.outHeight, s.eyeDistance, s.eyeRatioY)
    this.a = alignSide(faceA, s, dstEyes)
    this.b = alignSide(faceB, s, dstEyes)
    this.w = s.outWidth
    this.h = s.outHeight
    // Fixed topology from the midpoint mesh — stable across all t.
    const mid = new Float64Array(N_TOTAL * 2)
    for (let i = 0; i < N_TOTAL * 2; i++) mid[i] = (this.a.points[i] + this.b.points[i]) / 2
    this.tris = new Uint32Array(new Delaunator(mid).triangles)
  }

  renderFrame(t: number): ImageData {
    const w = this.w
    const h = this.h
    const mesh = new Float32Array(N_TOTAL * 2)
    for (let i = 0; i < N_TOTAL * 2; i++) {
      mesh[i] = this.a.points[i] * (1 - t) + this.b.points[i] * t
    }
    const engine = getWarpEngine()
    const pa = engine.warp(this.a.aligned, this.a.points, mesh, this.tris, w, h)
    const pb = engine.warp(this.b.aligned, this.b.points, mesh, this.tris, w, h)
    const out = new ImageData(w, h)
    const od = out.data
    for (let i = 0; i < od.length; i += 4) {
      const aa = pa[i + 3] / 255
      const ab = pb[i + 3] / 255
      const wa = aa * (1 - t)
      const wb = ab * t
      const tot = wa + wb
      if (tot > 1e-4) {
        od[i] = (pa[i] * wa + pb[i] * wb) / tot
        od[i + 1] = (pa[i + 1] * wa + pb[i + 1] * wb) / tot
        od[i + 2] = (pa[i + 2] * wa + pb[i + 2] * wb) / tot
        od[i + 3] = 255
      } else {
        od[i] = 18
        od[i + 1] = 21
        od[i + 2] = 26
        od[i + 3] = 255
      }
    }
    return out
  }
}

/** Ease-in-out sequence of t values, optionally boomerang (0→1→0). */
export function morphSchedule(frames: number, boomerang: boolean): number[] {
  const ease = (x: number) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2)
  const ts: number[] = []
  for (let i = 0; i < frames; i++) ts.push(ease(i / (frames - 1)))
  if (boomerang) return ts.concat([...ts].reverse())
  return ts
}
