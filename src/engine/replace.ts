// Warp-based face replace: pick the source image whose head pose best matches the target,
// warp its real texture onto the target's 478-pt mesh at native resolution, then mask,
// colour-match and composite it over the target. No neural models — deterministic and
// full-resolution, honouring "no loss in quality".

import Delaunator from 'delaunator'
import { getWarpEngine } from './warp'
import { rasterizeOval, insideFeather, makeCanvas } from './mask'
import { estimatePose, poseDistance } from './pose'
import { computeStats, applyTransfer } from './color'
import { yieldUI, type OnProgress } from './util'
import { N_LANDMARKS, type Face, type ReplaceSettings } from './types'

export interface ReplaceResult {
  imageData: ImageData
  sourceName: string
}

export async function computeReplace(
  sources: Face[],
  target: Face,
  s: ReplaceSettings,
  onProgress?: OnProgress,
): Promise<ReplaceResult> {
  // ---- validate ----
  const usable = sources.filter((f) => f.enabled && f.landmarks && !f.failed)
  if (usable.length === 0) throw new Error('Add at least one source face with a detected face')
  const tl = target.landmarks
  if (!tl) throw new Error('No face detected in the target image')

  const W = target.width,
    H = target.height // post-ingest native target size
  const dstPts = tl.points // 478*2, target pixel space

  // ---- pose bank: rank sources by pose match, tie-break by source-face resolution ----
  onProgress?.('Choosing the closest pose', 0.05)
  await yieldUI()
  const tPose = estimatePose(tl)
  const ranked = usable
    .map((f) => ({
      f,
      d: poseDistance(estimatePose(f.landmarks!), tPose),
      area: f.landmarks!.box.width * f.landmarks!.box.height,
    }))
    .sort((a, b) => a.d - b.d)
  // Among pose-equivalent shots (within +0.08 of the best), prefer the highest-resolution
  // face — more texture to sample means a sharper result.
  const near = ranked.filter((r) => r.d <= ranked[0].d + 0.08).sort((a, b) => b.area - a.area)
  const primary = near[0]
  const secondary = ranked.find((r) => r.f.id !== primary.f.id) ?? null
  const chosen = s.blendTopK === 2 && secondary ? [primary, secondary] : [primary]

  // ---- triangulate the TARGET's 478-pt mesh (face region only — no boundary points) ----
  // Triangulate on the destination points so output triangles are non-degenerate; the same
  // index list maps every source because all faces share MediaPipe's landmark indexing.
  const coords = new Float64Array(N_LANDMARKS * 2)
  for (let i = 0; i < N_LANDMARKS * 2; i++) coords[i] = dstPts[i]
  const tris = new Uint32Array(new Delaunator(coords).triangles)

  // ---- warp each chosen source onto the target mesh, at native target resolution ----
  // Raw bitmap + its own dims as srcW/srcH; mipmap for clean minification. No pre-alignment:
  // the mesh->mesh warp is the whole transform (aligning first would double-transform).
  const engine = getWarpEngine()
  const layers: Uint8ClampedArray[] = []
  for (let k = 0; k < chosen.length; k++) {
    onProgress?.(`Warping source ${k + 1}/${chosen.length}`, 0.15 + (k / chosen.length) * 0.3)
    await yieldUI()
    const { f } = chosen[k]
    layers.push(
      engine.warp(f.bitmap, f.landmarks!.points, dstPts, tris, W, H, {
        srcW: f.width,
        srcH: f.height,
        mipmap: true,
      }),
    )
  }
  const warped = layers[0]
  if (layers.length > 1) {
    for (let i = 0; i < warped.length; i += 4) {
      let r = 0,
        g = 0,
        b = 0,
        n = 0
      for (const L of layers)
        if (L[i + 3] > 8) {
          r += L[i]
          g += L[i + 1]
          b += L[i + 2]
          n++
        }
      if (n > 0) {
        warped[i] = r / n
        warped[i + 1] = g / n
        warped[i + 2] = b / n
        warped[i + 3] = 255
      } else {
        warped[i + 3] = 0
      }
    }
  }

  // ---- original target pixels (they are the composite background) ----
  onProgress?.('Building the face mask', 0.55)
  await yieldUI()
  const tc = makeCanvas(W, H)
  const tctx = tc.getContext('2d') as CanvasRenderingContext2D
  tctx.drawImage(target.bitmap as CanvasImageSource, 0, 0)
  const tgt = tctx.getImageData(0, 0, W, H)

  // ---- face mask: oval AND warp-coverage first, then inside-feather (seam-free ordering) ----
  const binary = rasterizeOval(dstPts, W, H, s.grow)
  for (let j = 0; j < binary.length; j++) {
    binary[j] = binary[j] < 0.5 || warped[j * 4 + 3] < 128 ? 0 : 1
  }
  const featherPx = Math.max(2, Math.round(s.feather * tl.box.width))
  const m = insideFeather(binary, W, H, featherPx)

  // ---- colour match: move the warped face toward the target's face-region stats ----
  onProgress?.('Matching colour', 0.75)
  await yieldUI()
  if (s.colorMatch > 0) {
    // computeStats ignores alpha<8, so masking via alpha yields face-region-only stats.
    const tFace = new Uint8ClampedArray(tgt.data) // copy — do not mutate tgt
    for (let j = 0; j < binary.length; j++) if (binary[j] === 0) tFace[j * 4 + 3] = 0
    const tStats = computeStats(tFace)
    const wStats = computeStats(warped) // warp alpha already gates the face
    if (tStats.count > 0 && wStats.count > 0) {
      applyTransfer(warped, wStats, tStats, s.colorMatch) // mutates `warped` in place
    }
  }

  // ---- composite: out = target*(1-m) + warped*m; fully opaque output ----
  onProgress?.('Compositing', 0.9)
  await yieldUI()
  const out = new ImageData(W, H)
  const od = out.data,
    td = tgt.data
  for (let j = 0; j < m.length; j++) {
    const o = j * 4
    const mv = m[j]
    od[o] = td[o] * (1 - mv) + warped[o] * mv
    od[o + 1] = td[o + 1] * (1 - mv) + warped[o + 1] * mv
    od[o + 2] = td[o + 2] * (1 - mv) + warped[o + 2] * mv
    od[o + 3] = 255
  }

  return { imageData: out, sourceName: chosen.map((c) => c.f.name).join(' + ') }
}
