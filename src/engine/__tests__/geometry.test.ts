import { describe, it, expect } from 'vitest'
import {
  similarityFrom2,
  canonicalEyes,
  transformLandmarks,
  boundaryPoints,
  N_TOTAL,
  type Pt,
} from '../geometry'
import { N_LANDMARKS, type Landmarks } from '../types'

type Affine = { a: number; b: number; c: number; d: number; tx: number; ty: number }
const apply = (m: Affine, p: Pt): Pt => [
  m.a * p[0] + m.b * p[1] + m.tx,
  m.c * p[0] + m.d * p[1] + m.ty,
]
const dist = (p: Pt, q: Pt) => Math.hypot(p[0] - q[0], p[1] - q[1])

describe('similarityFrom2', () => {
  it('maps the two anchor points exactly', () => {
    const src0: Pt = [1, 2]
    const src1: Pt = [5, 2]
    const dst0: Pt = [10, 10]
    const dst1: Pt = [10, 18]
    const m = similarityFrom2(src0, src1, dst0, dst1)
    const a0 = apply(m, src0)
    const a1 = apply(m, src1)
    expect(a0[0]).toBeCloseTo(10)
    expect(a0[1]).toBeCloseTo(10)
    expect(a1[0]).toBeCloseTo(10)
    expect(a1[1]).toBeCloseTo(18)
  })

  it('is a similarity: preserves distance ratios and applies a uniform scale', () => {
    // src vector (2,0) -> dst vector (0,4): a 90° rotation with a 2× scale.
    const m = similarityFrom2([0, 0], [2, 0], [3, 1], [3, 5])
    const A: Pt = [0, 0]
    const B: Pt = [4, 0]
    const C: Pt = [0, 3]
    const a = apply(m, A)
    const b = apply(m, B)
    const c = apply(m, C)
    expect(dist(a, b) / dist(a, c)).toBeCloseTo(dist(A, B) / dist(A, C))
    expect(dist(a, b) / dist(A, B)).toBeCloseTo(2)
  })
})

describe('canonicalEyes', () => {
  it('is symmetric about the width midline at a shared eye line', () => {
    const w = 800
    const h = 1000
    const { right, left } = canonicalEyes(w, h, 0.32, 2.6)
    expect(right[0] + left[0]).toBeCloseTo(w)
    expect(right[1]).toBeCloseTo(left[1])
    expect(left[0] - right[0]).toBeCloseTo(w * 0.32)
    expect(right[1]).toBeCloseTo(h / 2.6)
  })
})

describe('transformLandmarks', () => {
  const makeLandmarks = (): Landmarks => {
    const points = new Float32Array(N_LANDMARKS * 2)
    for (let i = 0; i < points.length; i++) points[i] = i
    return { points, box: { x: 0, y: 0, width: 1, height: 1 } }
  }

  it('leaves points unchanged under the identity transform', () => {
    const l = makeLandmarks()
    const out = transformLandmarks(l, { a: 1, b: 0, c: 0, d: 1, tx: 0, ty: 0 })
    expect(Array.from(out)).toEqual(Array.from(l.points))
  })

  it('translates every point and covers all landmarks', () => {
    const l = makeLandmarks()
    const out = transformLandmarks(l, { a: 1, b: 0, c: 0, d: 1, tx: 5, ty: -3 })
    expect(out.length).toBe(N_LANDMARKS * 2)
    expect(out[0]).toBeCloseTo(l.points[0] + 5)
    expect(out[1]).toBeCloseTo(l.points[1] - 3)
    expect(out[10]).toBeCloseTo(l.points[10] + 5)
    expect(out[11]).toBeCloseTo(l.points[11] - 3)
  })
})

describe('boundaryPoints', () => {
  it('returns the eight canvas-edge points', () => {
    expect(Array.from(boundaryPoints(100, 200))).toEqual([
      0, 0, 50, 0, 99, 0, 99, 100, 99, 199, 50, 199, 0, 199, 0, 100,
    ])
  })

  it('N_TOTAL accounts for the eight boundary points', () => {
    expect(N_TOTAL).toBe(N_LANDMARKS + 8)
  })
})
