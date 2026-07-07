import { describe, it, expect } from 'vitest'
import { blurMask, insideFeather, boxBlur } from '../mask'

describe('blurMask', () => {
  it('returns the input untouched for radius < 1', () => {
    const src = new Float32Array([0.2, 0.8, 0.5, 1])
    expect(blurMask(src, 2, 2, 0)).toBe(src)
  })

  it('leaves a constant field unchanged and within [0,1]', () => {
    const w = 8
    const h = 8
    const src = new Float32Array(w * h).fill(1)
    const out = blurMask(src, w, h, 2)
    for (const v of out) {
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThanOrEqual(1 + 1e-6)
      expect(v).toBeCloseTo(1)
    }
  })
})

describe('insideFeather', () => {
  it('keeps deep-inside pixels at 1 and far-outside at 0', () => {
    const w = 21
    const h = 21
    const mask = new Float32Array(w * h)
    for (let y = 5; y < 16; y++) for (let x = 5; x < 16; x++) mask[y * w + x] = 1
    const out = insideFeather(mask, w, h, 2)
    expect(out[10 * w + 10]).toBeCloseTo(1) // centre of the solid block
    expect(out[0]).toBe(0) // far corner, well outside
  })
})

describe('boxBlur', () => {
  it('preserves a constant opaque field and its dimensions', () => {
    const w = 5
    const h = 5
    const src = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < w * h; i++) src.set([100, 120, 140, 255], i * 4)
    const out = boxBlur(src, w, h, 4)
    expect(out.length).toBe(src.length)
    const c = (2 * w + 2) * 4 // centre pixel
    expect(out[c]).toBeCloseTo(100, 0)
    expect(out[c + 1]).toBeCloseTo(120, 0)
    expect(out[c + 2]).toBeCloseTo(140, 0)
    expect(out[c + 3]).toBe(255)
  })
})
