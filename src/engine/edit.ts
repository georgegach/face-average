// Mask-driven photo edits (FaceApp-style Retouch / Makeup / Hair / Scene tools).
// Every operation is a deterministic pixel op at native resolution, gated by soft masks
// from the 512px face parsing — no generative models, so nothing outside the edited
// pixels is ever touched and texture is preserved.
import { makeCanvas, boxBlur, insideFeather } from './mask'
import { classMask, CLS, type Parsing } from './parsing'
import { applyShape } from './shape'
import { IDX, type EditSettings, type Face } from './types'

const clamp255 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v)
const lum = (r: number, g: number, b: number) => 0.299 * r + 0.587 * g + 0.114 * b

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ]
}

// Native-res soft masks are expensive to build (sample + feather over megapixels), so
// cache them per (face, class-group, feather). Bitmaps never change per face id.
const maskCache = new Map<string, Float32Array>()

function getMask(
  faceId: string,
  parsing: Parsing,
  W: number,
  H: number,
  classes: readonly number[],
  featherPx: number,
): Float32Array {
  const key = `${faceId}:${classes.join(',')}:${featherPx}`
  let m = maskCache.get(key)
  if (!m) {
    m = classMask(parsing, W, H, classes, featherPx)
    maskCache.set(key, m)
    // Crude bound: masks are ~4MB each at 1600px; keep the cache from growing unbounded.
    if (maskCache.size > 24) {
      const first = maskCache.keys().next().value
      if (first) maskCache.delete(first)
    }
  }
  return m
}

export function computeEdit(
  face: Face,
  parsing: Parsing,
  s: EditSettings,
  base?: ImageData, // optional pre-processed frame (e.g. re-aged) to edit instead of the bitmap
): ImageData {
  if (!face.landmarks) throw new Error('No landmarks for the selected face')
  const W = face.width
  const H = face.height
  const faceW = face.landmarks.box.width
  const feather = Math.max(2, Math.round(faceW * 0.02))

  let out: ImageData
  if (base) {
    out = new ImageData(new Uint8ClampedArray(base.data), W, H)
  } else {
    const canvas = makeCanvas(W, H)
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D
    ctx.drawImage(face.bitmap as CanvasImageSource, 0, 0)
    out = ctx.getImageData(0, 0, W, H)
  }
  const d = out.data
  const pts = face.landmarks.points

  // ---- Retouch: skin smoothing (frequency separation, detail partially preserved) ----
  if (s.skinSmooth > 0) {
    const m = getMask(face.id, parsing, W, H, [CLS.skin, CLS.nose, CLS.neck], feather)
    const r = Math.max(2, Math.round(faceW * 0.025))
    const low = boxBlur(d, W, H, r)
    const k = s.skinSmooth * 0.85 // never fully flatten — keep pores
    for (let j = 0; j < m.length; j++) {
      const w = k * m[j]
      if (w <= 0) continue
      const o = j * 4
      d[o] = d[o] + (low[o] - d[o]) * w
      d[o + 1] = d[o + 1] + (low[o + 1] - d[o + 1]) * w
      d[o + 2] = d[o + 2] + (low[o + 2] - d[o + 2]) * w
    }
  }

  // ---- Retouch: teeth whitening (inner-mouth mask, gated to bright pixels) ----
  if (s.teethWhiten > 0) {
    const m = getMask(face.id, parsing, W, H, [CLS.mouth], Math.max(1, feather >> 1))
    for (let j = 0; j < m.length; j++) {
      if (m[j] <= 0) continue
      const o = j * 4
      const L = lum(d[o], d[o + 1], d[o + 2])
      const gate = Math.min(1, Math.max(0, (L - 70) / 60)) // skip tongue/shadow
      const w = s.teethWhiten * m[j] * gate
      if (w <= 0) continue
      // Reduce yellow (lift blue toward the warm channels), then brighten gently.
      const warm = Math.max(d[o], d[o + 1])
      d[o + 2] = clamp255(d[o + 2] + (warm - d[o + 2]) * 0.7 * w)
      d[o] = clamp255(d[o] + (255 - d[o]) * 0.22 * w)
      d[o + 1] = clamp255(d[o + 1] + (255 - d[o + 1]) * 0.22 * w)
      d[o + 2] = clamp255(d[o + 2] + (255 - d[o + 2]) * 0.22 * w)
    }
  }

  // ---- Makeup: lip tint (luminance-preserving colorize) ----
  if (s.lipColor && s.lipStrength > 0) {
    const [tr, tg, tb] = hexToRgb(s.lipColor)
    const tLum = Math.max(1, lum(tr, tg, tb))
    const m = getMask(face.id, parsing, W, H, [CLS.uLip, CLS.lLip], Math.max(1, feather >> 1))
    for (let j = 0; j < m.length; j++) {
      if (m[j] <= 0) continue
      const o = j * 4
      const w = s.lipStrength * m[j]
      const f = lum(d[o], d[o + 1], d[o + 2]) / tLum
      d[o] = clamp255(d[o] + (tr * f - d[o]) * w)
      d[o + 1] = clamp255(d[o + 1] + (tg * f - d[o + 1]) * w)
      d[o + 2] = clamp255(d[o + 2] + (tb * f - d[o + 2]) * w)
    }
  }

  // ---- Makeup: brow definition (darken + slight contrast in the brow mask) ----
  if (s.browDefine > 0) {
    const m = getMask(face.id, parsing, W, H, [CLS.lBrow, CLS.rBrow], Math.max(1, feather >> 1))
    for (let j = 0; j < m.length; j++) {
      if (m[j] <= 0) continue
      const o = j * 4
      const w = 1 - 0.32 * s.browDefine * m[j]
      d[o] *= w
      d[o + 1] *= w
      d[o + 2] *= w
    }
  }

  // ---- Makeup: blush (gaussian falloff around the cheek landmarks) ----
  if (s.blush > 0) {
    const rose: [number, number, number] = [226, 106, 122]
    const radius = faceW * 0.16
    const sigma2 = 2 * (radius * 0.7) * (radius * 0.7)
    for (const idx of [205, 425]) {
      // MediaPipe cheek-center landmarks
      const cxp = pts[idx * 2]
      const cyp = pts[idx * 2 + 1]
      const x0 = Math.max(0, Math.floor(cxp - radius * 1.6))
      const x1 = Math.min(W - 1, Math.ceil(cxp + radius * 1.6))
      const y0 = Math.max(0, Math.floor(cyp - radius * 1.6))
      const y1 = Math.min(H - 1, Math.ceil(cyp + radius * 1.6))
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const dx = x - cxp
          const dy = y - cyp
          const fall = Math.exp(-(dx * dx + dy * dy) / sigma2)
          const w = s.blush * 0.38 * fall
          if (w < 0.01) continue
          const o = (y * W + x) * 4
          const f = lum(d[o], d[o + 1], d[o + 2]) / lum(...rose)
          d[o] = clamp255(d[o] + (rose[0] * f - d[o]) * w)
          d[o + 1] = clamp255(d[o + 1] + (rose[1] * f - d[o + 1]) * w)
          d[o + 2] = clamp255(d[o + 2] + (rose[2] * f - d[o + 2]) * w)
        }
      }
    }
  }

  // ---- Makeup: eye colour (iris circles from the refined iris landmarks) ----
  if (s.eyeColor && s.eyeStrength > 0) {
    const [tr, tg, tb] = hexToRgb(s.eyeColor)
    const tLum = Math.max(1, lum(tr, tg, tb))
    for (const [center, ring] of [
      [IDX.rightIris, [469, 470, 471, 472]],
      [IDX.leftIris, [474, 475, 476, 477]],
    ] as [number, number[]][]) {
      const cxp = pts[center * 2]
      const cyp = pts[center * 2 + 1]
      let radius = 0
      for (const rI of ring) {
        const dx = pts[rI * 2] - cxp
        const dy = pts[rI * 2 + 1] - cyp
        radius += Math.hypot(dx, dy)
      }
      radius /= ring.length
      const rMax = radius * 1.1
      const x0 = Math.max(0, Math.floor(cxp - rMax))
      const x1 = Math.min(W - 1, Math.ceil(cxp + rMax))
      const y0 = Math.max(0, Math.floor(cyp - rMax))
      const y1 = Math.min(H - 1, Math.ceil(cyp + rMax))
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const dist = Math.hypot(x - cxp, y - cyp)
          if (dist > rMax) continue
          const o = (y * W + x) * 4
          const L = lum(d[o], d[o + 1], d[o + 2])
          // Skip the pupil (dark) and specular highlights (bright); soft gates.
          const gate =
            Math.min(1, Math.max(0, (L - 28) / 40)) * Math.min(1, Math.max(0, (215 - L) / 40))
          const fall = Math.min(1, Math.max(0, (rMax - dist) / (radius * 0.3)))
          const w = s.eyeStrength * gate * fall
          if (w <= 0.01) continue
          const f = L / tLum
          d[o] = clamp255(d[o] + (tr * f - d[o]) * w)
          d[o + 1] = clamp255(d[o + 1] + (tg * f - d[o + 1]) * w)
          d[o + 2] = clamp255(d[o + 2] + (tb * f - d[o + 2]) * w)
        }
      }
    }
  }

  // ---- Hair colour (texture-preserving recolor with lift for light targets) ----
  if (s.hairColor && s.hairStrength > 0) {
    const [tr, tg, tb] = hexToRgb(s.hairColor)
    const tLum = Math.max(1, lum(tr, tg, tb))
    const m = getMask(face.id, parsing, W, H, [CLS.hair], feather)
    for (let j = 0; j < m.length; j++) {
      if (m[j] <= 0) continue
      const o = j * 4
      const w = s.hairStrength * m[j]
      const L = lum(d[o], d[o + 1], d[o + 2])
      // Lift dark hair toward light targets so blonde/silver reads; keeps highlights.
      const effL = L + Math.max(0, tLum - L) * 0.45
      const f = effL / tLum
      d[o] = clamp255(d[o] + (tr * f - d[o]) * w)
      d[o + 1] = clamp255(d[o + 1] + (tg * f - d[o + 1]) * w)
      d[o + 2] = clamp255(d[o + 2] + (tb * f - d[o + 2]) * w)
    }
  }

  // ---- Scene: background bokeh / studio ----
  if (s.background !== 'none' && s.backgroundStrength > 0) {
    // Foreground = every non-background class; inside-feather it so the blend ramp sits
    // inside the person and no background halo bleeds onto the subject.
    const fgClasses: number[] = []
    for (let c = 1; c <= 18; c++) fgClasses.push(c)
    const fgRaw = getMask(face.id, parsing, W, H, fgClasses, 0)
    const fgKey = `${face.id}:fg-inner:${feather}`
    let fgInner = maskCache.get(fgKey)
    if (!fgInner) {
      fgInner = insideFeather(fgRaw, W, H, Math.round(feather * 1.5))
      maskCache.set(fgKey, fgInner)
    }
    if (s.background === 'bokeh') {
      const r = Math.round(W * (0.015 + 0.045 * s.backgroundStrength))
      const blurred = boxBlur(d, W, H, r)
      for (let j = 0; j < fgInner.length; j++) {
        const bw = 1 - fgInner[j]
        if (bw <= 0) continue
        const o = j * 4
        d[o] = d[o] + (blurred[o] - d[o]) * bw
        d[o + 1] = d[o + 1] + (blurred[o + 1] - d[o + 1]) * bw
        d[o + 2] = d[o + 2] + (blurred[o + 2] - d[o + 2]) * bw
      }
    } else {
      // studio: subtle vertical gradient in the app's dark-studio palette
      for (let y = 0; y < H; y++) {
        const t = y / H
        const br = 30 - 12 * t
        const bg = 35 - 13 * t
        const bb = 44 - 16 * t
        for (let x = 0; x < W; x++) {
          const j = y * W + x
          const bw = (1 - fgInner[j]) * s.backgroundStrength
          if (bw <= 0) continue
          const o = j * 4
          d[o] = d[o] + (br - d[o]) * bw
          d[o + 1] = d[o + 1] + (bg - d[o + 1]) * bw
          d[o + 2] = d[o + 2] + (bb - d[o + 2]) * bw
        }
      }
    }
  }

  // ---- Scene: vignette ----
  if (s.vignette > 0) {
    const cx = W / 2
    const cy = H / 2
    const maxD = Math.hypot(cx, cy)
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const t = (Math.hypot(x - cx, y - cy) / maxD - 0.55) / 0.45
        if (t <= 0) continue
        const v = 1 - Math.min(1, t) * Math.min(1, t) * 0.55 * s.vignette
        const o = (y * W + x) * 4
        d[o] *= v
        d[o + 1] *= v
        d[o + 2] *= v
      }
    }
  }

  // ---- Shape warps last: pixel edits above use masks aligned to the original bitmap;
  // the warp then moves the (already-edited) pixels in one geometric pass. ----
  return applyShape(out, face, s)
}
