import { describe, it, expect } from 'vitest'
import { computeStats, applyTransfer, averageStats } from '../color'

type RGBA = [number, number, number, number]
function px(pixels: RGBA[]): Uint8ClampedArray {
  const arr = new Uint8ClampedArray(pixels.length * 4)
  pixels.forEach((p, i) => arr.set(p, i * 4))
  return arr
}

describe('computeStats', () => {
  it('computes per-channel mean and std over opaque pixels', () => {
    const s = computeStats(px([
      [10, 20, 30, 255],
      [30, 40, 50, 255],
    ]))
    expect(s.count).toBe(2)
    expect(s.mean).toEqual([20, 30, 40])
    expect(s.std[0]).toBeCloseTo(10)
    expect(s.std[1]).toBeCloseTo(10)
    expect(s.std[2]).toBeCloseTo(10)
  })

  it('ignores near-transparent pixels (alpha < 8)', () => {
    const s = computeStats(px([
      [10, 20, 30, 255],
      [30, 40, 50, 255],
      [200, 200, 200, 4],
    ]))
    expect(s.count).toBe(2)
    expect(s.mean).toEqual([20, 30, 40])
  })

  it('returns a neutral result when nothing is opaque', () => {
    const s = computeStats(px([[0, 0, 0, 0]]))
    expect(s.count).toBe(0)
    expect(s.std).toEqual([1, 1, 1])
  })
})

describe('applyTransfer', () => {
  it('is a no-op when source and target stats match', () => {
    const data = px([[10, 20, 30, 255], [30, 40, 50, 255]])
    const before = Array.from(data)
    const stats = computeStats(data)
    applyTransfer(data, stats, stats, 1)
    expect(Array.from(data)).toEqual(before)
  })

  it('is a no-op at strength 0 regardless of target', () => {
    const data = px([[10, 20, 30, 255], [30, 40, 50, 255]])
    const before = Array.from(data)
    const src = computeStats(data)
    const target = {
      mean: [200, 200, 200] as [number, number, number],
      std: [50, 50, 50] as [number, number, number],
      count: 5,
    }
    applyTransfer(data, src, target, 0)
    expect(Array.from(data)).toEqual(before)
  })
})

describe('averageStats', () => {
  it('averages matching stats to the same mean/std and counts contributors', () => {
    const s = computeStats(px([[10, 20, 30, 255], [30, 40, 50, 255]]))
    const avg = averageStats([s, s, s])
    expect(avg.mean).toEqual(s.mean)
    expect(avg.std[0]).toBeCloseTo(s.std[0])
    expect(avg.count).toBe(3)
  })

  it('skips zero-count entries', () => {
    const s = computeStats(px([[10, 20, 30, 255], [30, 40, 50, 255]]))
    const empty = computeStats(px([[0, 0, 0, 0]]))
    const avg = averageStats([s, empty])
    expect(avg.count).toBe(1)
    expect(avg.mean).toEqual(s.mean)
  })
})
