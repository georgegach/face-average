import { describe, it, expect } from 'vitest'
import { morphSchedule } from '../morph'

describe('morphSchedule', () => {
  it('runs an eased, monotonic 0->1 ramp for a one-way pass', () => {
    const ts = morphSchedule(10, false)
    expect(ts).toHaveLength(10)
    expect(ts[0]).toBeCloseTo(0)
    expect(ts[ts.length - 1]).toBeCloseTo(1)
    for (let i = 1; i < ts.length; i++) {
      expect(ts[i]).toBeGreaterThanOrEqual(ts[i - 1])
    }
  })

  it('is a palindrome that returns to 0 for a boomerang pass', () => {
    const ts = morphSchedule(8, true)
    expect(ts).toHaveLength(16)
    expect(ts[0]).toBeCloseTo(0)
    expect(ts[ts.length - 1]).toBeCloseTo(0)
    expect(Math.max(...ts)).toBeCloseTo(1)
    for (let i = 0; i < ts.length; i++) {
      expect(ts[i]).toBeCloseTo(ts[ts.length - 1 - i])
    }
  })
})
