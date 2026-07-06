// Parametric shape warps (smile, eye size, nose/face slim, hair volume) built as
// displacement fields over the 478-pt mesh, applied as one GPU piecewise-affine warp.
// Deterministic and native-res: geometry moves, texture is only resampled once.
import Delaunator from 'delaunator'
import { boundaryPoints, irisCenters } from './geometry'
import { FACE_OVAL, makeCanvas } from './mask'
import { getWarpEngine } from './warp'
import { N_LANDMARKS, IDX, type Face, type EditSettings } from './types'

// Upper half of the face-oval ring — anchors for the synthetic hair ring that lets the
// warp reach pixels above the forehead (there are no landmarks in the hair itself).
const HAIR_ANCHORS = [10, 338, 297, 332, 284, 251, 389, 356, 162, 21, 54, 103, 67, 109, 127]

export function hasShapeEdits(s: EditSettings): boolean {
  return (
    s.smile !== 0 || s.eyeSize !== 0 || s.noseSlim > 0 || s.faceSlim > 0 || s.hairVolume > 0
  )
}

const smooth = (t: number) => {
  const c = t < 0 ? 0 : t > 1 ? 1 : t
  return c * c * (3 - 2 * c)
}

/**
 * Apply the shape tools to an already-edited image (pixel ops happen before the warp so
 * parsing masks stay aligned with the original bitmap).
 */
export function applyShape(img: ImageData, face: Face, s: EditSettings): ImageData {
  if (!face.landmarks || !hasShapeEdits(s)) return img
  const W = img.width
  const H = img.height
  const pts = face.landmarks.points
  const faceW = face.landmarks.box.width
  const faceH = face.landmarks.box.height

  // Extended mesh: 478 landmarks + hair ring + canvas boundary. src = identity positions.
  const nHair = HAIR_ANCHORS.length
  const nTotal = N_LANDMARKS + nHair + 8
  const src = new Float32Array(nTotal * 2)
  src.set(pts.subarray(0, N_LANDMARKS * 2), 0)

  // Face centroid (from the oval ring) — hair ring extends outward from it.
  let cx = 0
  let cy = 0
  for (const i of FACE_OVAL) {
    cx += pts[i * 2]
    cy += pts[i * 2 + 1]
  }
  cx /= FACE_OVAL.length
  cy /= FACE_OVAL.length

  for (let k = 0; k < nHair; k++) {
    const a = HAIR_ANCHORS[k]
    const off = (N_LANDMARKS + k) * 2
    src[off] = cx + (pts[a * 2] - cx) * 1.5
    src[off + 1] = cy + (pts[a * 2 + 1] - cy) * 1.5
  }
  src.set(boundaryPoints(W, H), (N_LANDMARKS + nHair) * 2)

  const dst = new Float32Array(src) // displaced copy; boundary stays pinned

  // ---- smile: lift/drop the mouth corners with gaussian falloff over the mouth ----
  if (s.smile !== 0) {
    const corners = [61, 291] // subject-right and subject-left mouth corners
    const mouthW = Math.hypot(pts[291 * 2] - pts[61 * 2], pts[291 * 2 + 1] - pts[61 * 2 + 1])
    const sigma2 = 2 * (mouthW * 0.55) * (mouthW * 0.55)
    const dy = -s.smile * faceW * 0.035
    for (const c of corners) {
      const ccx = pts[c * 2]
      const ccy = pts[c * 2 + 1]
      const dxDir = c === 61 ? -1 : 1 // corners also travel slightly outward
      const dx = dxDir * s.smile * faceW * 0.018
      for (let i = 0; i < N_LANDMARKS; i++) {
        const px = pts[i * 2]
        const py = pts[i * 2 + 1]
        const d2 = (px - ccx) * (px - ccx) + (py - ccy) * (py - ccy)
        const w = Math.exp(-d2 / sigma2)
        if (w < 0.01) continue
        dst[i * 2] += dx * w
        dst[i * 2 + 1] += dy * w
      }
    }
  }

  // ---- eye size: radial scale around each iris with smooth falloff ----
  if (s.eyeSize !== 0) {
    const { right, left } = irisCenters(face.landmarks)
    const eyeW = Math.hypot(pts[133 * 2] - pts[33 * 2], pts[133 * 2 + 1] - pts[33 * 2 + 1])
    const R = eyeW * 1.5
    const k = s.eyeSize * 0.22
    for (const [ecx, ecy] of [right, left]) {
      for (let i = 0; i < N_LANDMARKS; i++) {
        const px = pts[i * 2]
        const py = pts[i * 2 + 1]
        const dist = Math.hypot(px - ecx, py - ecy)
        if (dist >= R || dist < 1e-3) continue
        const w = 1 - smooth(dist / R)
        dst[i * 2] += (px - ecx) * k * w
        dst[i * 2 + 1] += (py - ecy) * k * w
      }
    }
  }

  // ---- nose slim: pull the nose flanks toward the vertical nose axis ----
  if (s.noseSlim > 0) {
    const axX = pts[IDX.noseTip * 2]
    const noseTop = pts[168 * 2 + 1] // bridge between the eyes
    const noseBot = pts[2 * 2 + 1] // just under the nose
    const span = Math.max(1, noseBot - noseTop)
    const halfW = faceW * 0.22
    for (let i = 0; i < N_LANDMARKS; i++) {
      const px = pts[i * 2]
      const py = pts[i * 2 + 1]
      const dxAxis = px - axX
      const ax = Math.abs(dxAxis)
      if (ax > halfW) continue
      const vy = (py - noseTop) / span
      if (vy < -0.15 || vy > 1.25) continue
      const wy = smooth(1 - Math.abs(vy - 0.6) / 0.85)
      const wx = 1 - smooth(ax / halfW)
      dst[i * 2] -= dxAxis * s.noseSlim * 0.16 * wy * wx
    }
  }

  // ---- face slim: pull the jaw/cheek silhouette toward the face's vertical axis ----
  if (s.faceSlim > 0) {
    const eyeY = (pts[IDX.rightIris * 2 + 1] + pts[IDX.leftIris * 2 + 1]) / 2
    const chinY = pts[IDX.chin * 2 + 1]
    const span = Math.max(1, chinY - eyeY)
    for (let i = 0; i < N_LANDMARKS; i++) {
      const px = pts[i * 2]
      const py = pts[i * 2 + 1]
      const vy = (py - eyeY) / span
      if (vy < 0.05) continue // never move the eyes/forehead
      const wy = smooth(Math.min(1, vy)) // strongest at the jaw
      const ax = Math.abs(px - cx) / (faceW * 0.55)
      if (ax > 1.4) continue
      const wx = smooth(Math.min(1, ax)) // strongest at the silhouette, zero at the axis
      dst[i * 2] -= (px - cx) * s.faceSlim * 0.1 * wy * wx
    }
  }

  // ---- hair volume: push the synthetic hair ring outward ----
  if (s.hairVolume > 0) {
    const k = s.hairVolume * 0.16
    for (let j = 0; j < nHair; j++) {
      const off = (N_LANDMARKS + j) * 2
      dst[off] += (src[off] - cx) * k
      dst[off + 1] += (src[off + 1] - cy) * k
      // Also nudge the corresponding oval anchor slightly so the scalp follows.
      const a = HAIR_ANCHORS[j]
      const ay = pts[a * 2 + 1]
      if (ay < cy - faceH * 0.2) {
        dst[a * 2] += (pts[a * 2] - cx) * k * 0.35
        dst[a * 2 + 1] += (ay - cy) * k * 0.35
      }
    }
  }

  // Triangulate over the destination mesh (matches the averaging pipeline's convention).
  const coords = new Float64Array(nTotal * 2)
  for (let i = 0; i < nTotal * 2; i++) coords[i] = dst[i]
  const tris = new Uint32Array(new Delaunator(coords).triangles)

  const canvas = makeCanvas(W, H)
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D
  ctx.putImageData(img, 0, 0)
  const px = getWarpEngine().warp(canvas, src, dst, tris, W, H)
  return new ImageData(px, W, H)
}
