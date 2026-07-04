// Simplified Reinhard color transfer: match each face's per-channel mean/std to
// a group target so inputs shot under different lighting don't muddy the average.

export interface ChannelStats {
  mean: [number, number, number]
  std: [number, number, number]
  count: number
}

export function computeStats(px: Uint8ClampedArray): ChannelStats {
  let n = 0
  const sum = [0, 0, 0]
  const sq = [0, 0, 0]
  for (let i = 0; i < px.length; i += 4) {
    if (px[i + 3] < 8) continue // ignore uncovered pixels
    for (let c = 0; c < 3; c++) {
      const v = px[i + c]
      sum[c] += v
      sq[c] += v * v
    }
    n++
  }
  if (n === 0) return { mean: [0, 0, 0], std: [1, 1, 1], count: 0 }
  const mean: [number, number, number] = [sum[0] / n, sum[1] / n, sum[2] / n]
  const std: [number, number, number] = [
    Math.sqrt(Math.max(1e-6, sq[0] / n - mean[0] * mean[0])),
    Math.sqrt(Math.max(1e-6, sq[1] / n - mean[1] * mean[1])),
    Math.sqrt(Math.max(1e-6, sq[2] / n - mean[2] * mean[2])),
  ]
  return { mean, std, count: n }
}

/** Shift `px` in place so its stats move toward `target` by `strength` (0..1). */
export function applyTransfer(
  px: Uint8ClampedArray,
  src: ChannelStats,
  target: ChannelStats,
  strength = 0.7,
) {
  for (let i = 0; i < px.length; i += 4) {
    if (px[i + 3] < 8) continue
    for (let c = 0; c < 3; c++) {
      const s = target.std[c] / src.std[c]
      // Full Reinhard mapping, then blend toward the original by `strength`.
      const mapped = (px[i + c] - src.mean[c]) * s + target.mean[c]
      const v = px[i + c] + (mapped - px[i + c]) * strength
      px[i + c] = v < 0 ? 0 : v > 255 ? 255 : v
    }
  }
}

export function averageStats(list: ChannelStats[]): ChannelStats {
  const w = list.filter((s) => s.count > 0)
  if (!w.length) return { mean: [0, 0, 0], std: [1, 1, 1], count: 0 }
  const mean: [number, number, number] = [0, 0, 0]
  const std: [number, number, number] = [0, 0, 0]
  for (const s of w)
    for (let c = 0; c < 3; c++) {
      mean[c] += s.mean[c]
      std[c] += s.std[c]
    }
  for (let c = 0; c < 3; c++) {
    mean[c] /= w.length
    std[c] /= w.length
  }
  return { mean, std, count: w.length }
}
