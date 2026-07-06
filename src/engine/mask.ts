// Shared canvas/mask helpers used by both the averaging and face-replace pipelines.

export function makeCanvas(w: number, h: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h)
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  return c
}

// FaceMesh outer silhouette loop (FACEMESH_FACE_OVAL), ordered.
export const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
  148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

/**
 * Hard 0..1 face mask (length w*h) from the 478-pt mesh's face-oval ring, in the coordinate
 * space of `pts` (interleaved x,y). The polygon is grown from its centroid by `grow`.
 */
export function rasterizeOval(
  pts: Float32Array,
  w: number,
  h: number,
  grow: number,
): Float32Array {
  let cx = 0,
    cy = 0
  for (const i of FACE_OVAL) {
    cx += pts[i * 2]
    cy += pts[i * 2 + 1]
  }
  cx /= FACE_OVAL.length
  cy /= FACE_OVAL.length

  const mc = makeCanvas(w, h)
  const mctx = mc.getContext('2d') as CanvasRenderingContext2D
  mctx.clearRect(0, 0, w, h)
  mctx.fillStyle = '#fff'
  mctx.beginPath()
  FACE_OVAL.forEach((idx, k) => {
    const x = cx + (pts[idx * 2] - cx) * grow
    const y = cy + (pts[idx * 2 + 1] - cy) * grow
    if (k === 0) mctx.moveTo(x, y)
    else mctx.lineTo(x, y)
  })
  mctx.closePath()
  mctx.fill()

  const raw = mctx.getImageData(0, 0, w, h).data
  const mask = new Float32Array(w * h)
  for (let j = 0; j < mask.length; j++) mask[j] = raw[j * 4 + 3] / 255
  return mask
}

/**
 * Feather `mask` inward: 0 at the region boundary rising to 1 at ~r px inside.
 *
 * A plain symmetric blur puts half the ramp *outside* the region; when the mask is later
 * clipped by the warp's coverage hull, that outside half is cut off and leaves a hard seam
 * (~0.5 at the edge, 0 one pixel further). Blurring then remapping [0.5,1] -> [0,1] keeps the
 * entire ramp inside the region, so it can be clipped with no seam.
 */
export function insideFeather(
  mask: Float32Array,
  w: number,
  h: number,
  r: number,
): Float32Array {
  const b = blurMask(mask, w, h, Math.max(1, Math.round(r)))
  const out = new Float32Array(mask.length)
  for (let j = 0; j < out.length; j++) {
    const v = (b[j] - 0.5) * 2
    out[j] = v < 0 ? 0 : v > 1 ? 1 : v
  }
  return out
}

export function blurMask(src: Float32Array, w: number, h: number, r: number): Float32Array {
  if (r < 1) return src
  const tmp = new Float32Array(w * h)
  const inv = 1 / (2 * r + 1)
  for (let y = 0; y < h; y++) {
    let acc = 0
    for (let x = -r; x <= r; x++) acc += src[y * w + Math.min(w - 1, Math.max(0, x))]
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = acc * inv
      const add = src[y * w + Math.min(w - 1, x + r + 1)]
      const sub = src[y * w + Math.max(0, x - r)]
      acc += add - sub
    }
  }
  const out = new Float32Array(w * h)
  for (let x = 0; x < w; x++) {
    let acc = 0
    for (let y = -r; y <= r; y++) acc += tmp[Math.min(h - 1, Math.max(0, y)) * w + x]
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc * inv
      const add = tmp[Math.min(h - 1, y + r + 1) * w + x]
      const sub = tmp[Math.max(0, y - r) * w + x]
      acc += add - sub
    }
  }
  return out
}

export function boxBlur(
  src: Uint8ClampedArray,
  w: number,
  h: number,
  r: number,
): Uint8ClampedArray {
  // Cheap separable box blur that ignores transparent source pixels.
  const tmp = new Float32Array(w * h * 3)
  const tw = new Float32Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x
      let r0 = 0,
        g0 = 0,
        b0 = 0,
        wsum = 0
      for (let dx = -r; dx <= r; dx += 4) {
        const xx = x + dx
        if (xx < 0 || xx >= w) continue
        const si = (y * w + xx) * 4
        if (src[si + 3] === 0) continue
        r0 += src[si]
        g0 += src[si + 1]
        b0 += src[si + 2]
        wsum++
      }
      tmp[idx * 3] = r0
      tmp[idx * 3 + 1] = g0
      tmp[idx * 3 + 2] = b0
      tw[idx] = wsum
    }
  }
  const out = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x
      let r0 = 0,
        g0 = 0,
        b0 = 0,
        wsum = 0
      for (let dy = -r; dy <= r; dy += 4) {
        const yy = y + dy
        if (yy < 0 || yy >= h) continue
        const si = yy * w + x
        if (tw[si] === 0) continue
        r0 += tmp[si * 3]
        g0 += tmp[si * 3 + 1]
        b0 += tmp[si * 3 + 2]
        wsum += tw[si]
      }
      const o = idx * 4
      if (wsum > 0) {
        out[o] = r0 / wsum
        out[o + 1] = g0 / wsum
        out[o + 2] = b0 / wsum
        out[o + 3] = 255
      }
    }
  }
  return out
}
