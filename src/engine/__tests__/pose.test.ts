import { describe, it, expect } from 'vitest'
import { estimatePose, poseDistance } from '../pose'
import { N_LANDMARKS, IDX, type Landmarks } from '../types'

function landmarksFrom(coords: Record<number, [number, number]>): Landmarks {
  const points = new Float32Array(N_LANDMARKS * 2)
  for (const [i, [x, y]] of Object.entries(coords)) {
    const idx = Number(i)
    points[idx * 2] = x
    points[idx * 2 + 1] = y
  }
  return { points, box: { x: 0, y: 0, width: 1, height: 1 } }
}

const frontal = landmarksFrom({
  [IDX.rightIris]: [40, 50],
  [IDX.leftIris]: [60, 50],
  [IDX.noseTip]: [50, 60],
  [IDX.chin]: [50, 70],
  [IDX.rightFace]: [30, 55],
  [IDX.leftFace]: [70, 55],
})

describe('estimatePose', () => {
  it('reads ~zero yaw/pitch/roll for a level frontal face', () => {
    const p = estimatePose(frontal)
    expect(p.roll).toBeCloseTo(0)
    expect(p.yaw).toBeCloseTo(0)
    expect(p.pitch).toBeCloseTo(0)
  })

  it('detects head roll from a tilted iris line', () => {
    const tilted = landmarksFrom({
      [IDX.rightIris]: [40, 48],
      [IDX.leftIris]: [60, 52],
      [IDX.noseTip]: [50, 60],
      [IDX.chin]: [50, 70],
      [IDX.rightFace]: [30, 55],
      [IDX.leftFace]: [70, 55],
    })
    const p = estimatePose(tilted)
    expect(p.roll).toBeCloseTo(Math.atan2(4, 20))
    expect(p.roll).toBeGreaterThan(0)
  })

  it('detects yaw when the nose shifts toward one cheek', () => {
    const turned = landmarksFrom({
      [IDX.rightIris]: [40, 50],
      [IDX.leftIris]: [60, 50],
      [IDX.noseTip]: [45, 60],
      [IDX.chin]: [50, 70],
      [IDX.rightFace]: [30, 55],
      [IDX.leftFace]: [70, 55],
    })
    expect(estimatePose(turned).yaw).toBeGreaterThan(0)
  })
})

describe('poseDistance', () => {
  it('is zero for identical poses', () => {
    const p = estimatePose(frontal)
    expect(poseDistance(p, p)).toBe(0)
  })

  it('is symmetric', () => {
    const a = { yaw: 0.2, pitch: -0.1, roll: 0.3 }
    const b = { yaw: -0.1, pitch: 0.4, roll: -0.2 }
    expect(poseDistance(a, b)).toBeCloseTo(poseDistance(b, a))
  })

  it('weights roll at a quarter of yaw', () => {
    const base = { yaw: 0, pitch: 0, roll: 0 }
    expect(poseDistance(base, { yaw: 2, pitch: 0, roll: 0 })).toBeCloseTo(2)
    expect(poseDistance(base, { yaw: 0, pitch: 0, roll: 2 })).toBeCloseTo(1) // sqrt(0.25 * 4)
  })
})
