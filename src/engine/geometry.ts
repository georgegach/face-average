import { N_LANDMARKS, IDX, type Landmarks } from './types'

export interface Affine {
  // maps (x,y) -> (a*x + b*y + tx, c*x + d*y + ty)
  a: number
  b: number
  c: number
  d: number
  tx: number
  ty: number
}

export type Pt = [number, number]

/** Exact similarity (scale + rotation + translation) from two point pairs. */
export function similarityFrom2(src0: Pt, src1: Pt, dst0: Pt, dst1: Pt): Affine {
  const sdx = src1[0] - src0[0]
  const sdy = src1[1] - src0[1]
  const ddx = dst1[0] - dst0[0]
  const ddy = dst1[1] - dst0[1]
  const srcLen2 = sdx * sdx + sdy * sdy || 1e-6
  // Solve for scaled rotation (a,b) s.t. R*src_vec = dst_vec.
  const a = (sdx * ddx + sdy * ddy) / srcLen2
  const b = (sdx * ddy - sdy * ddx) / srcLen2
  // [a -b; b a] is a scaled rotation.
  const tx = dst0[0] - (a * src0[0] - b * src0[1])
  const ty = dst0[1] - (b * src0[0] + a * src0[1])
  return { a, b: -b, c: b, d: a, tx, ty }
}

export function applyAffine(m: Affine, p: Pt): Pt {
  return [m.a * p[0] + m.b * p[1] + m.tx, m.c * p[0] + m.d * p[1] + m.ty]
}

export function irisCenters(l: Landmarks): { right: Pt; left: Pt } {
  const p = l.points
  return {
    right: [p[IDX.rightIris * 2], p[IDX.rightIris * 2 + 1]],
    left: [p[IDX.leftIris * 2], p[IDX.leftIris * 2 + 1]],
  }
}

/** Canonical destination eye positions in output space. */
export function canonicalEyes(w: number, h: number, eyeDistance: number, eyeRatioY: number) {
  const cx = w / 2
  const cy = h / eyeRatioY
  const half = (w * eyeDistance) / 2
  // right iris on the left of the image (subject's right), left iris on the right.
  const right: Pt = [cx - half, cy]
  const left: Pt = [cx + half, cy]
  return { right, left }
}

/** Transform all landmark points by an affine; returns interleaved Float32Array. */
export function transformLandmarks(l: Landmarks, m: Affine): Float32Array {
  const out = new Float32Array(N_LANDMARKS * 2)
  const p = l.points
  for (let i = 0; i < N_LANDMARKS; i++) {
    const x = p[i * 2]
    const y = p[i * 2 + 1]
    out[i * 2] = m.a * x + m.b * y + m.tx
    out[i * 2 + 1] = m.c * x + m.d * y + m.ty
  }
  return out
}

/** 8 canvas boundary points appended to every mesh (corners + edge midpoints). */
export function boundaryPoints(w: number, h: number): Float32Array {
  const b = [
    0, 0,
    w / 2, 0,
    w - 1, 0,
    w - 1, h / 2,
    w - 1, h - 1,
    w / 2, h - 1,
    0, h - 1,
    0, h / 2,
  ]
  return new Float32Array(b)
}

export const N_BOUNDARY = 8
export const N_TOTAL = N_LANDMARKS + N_BOUNDARY
