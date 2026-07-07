import { describe, it, expect } from 'vitest'
import { hasShapeEdits } from '../shape'
import { DEFAULT_EDIT_SETTINGS } from '../types'

describe('hasShapeEdits', () => {
  it('is false for the default settings', () => {
    expect(hasShapeEdits(DEFAULT_EDIT_SETTINGS)).toBe(false)
  })

  it('is true when any shape control is set', () => {
    const cases = [
      { smile: 0.5 },
      { smile: -0.3 },
      { eyeSize: 0.4 },
      { eyeSize: -0.2 },
      { noseSlim: 0.1 },
      { faceSlim: 0.6 },
      { hairVolume: 0.2 },
    ]
    for (const patch of cases) {
      expect(hasShapeEdits({ ...DEFAULT_EDIT_SETTINGS, ...patch })).toBe(true)
    }
  })

  it('ignores non-shape controls', () => {
    expect(
      hasShapeEdits({ ...DEFAULT_EDIT_SETTINGS, skinSmooth: 1, lipColor: '#ffffff', vignette: 0.5 }),
    ).toBe(false)
  })
})
