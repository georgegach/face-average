import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision'
import { MODELS } from './models'
import { fetchWithProgress } from './download'
import { progressChannel } from './util'
import type { Landmarks } from './types'

// MediaPipe's WASM loader relies on importScripts, which is unavailable in ES
// module workers, so the landmarker runs on the main thread. Detection is fast
// (CPU delegate) and infrequent enough that this keeps the UI responsive.
let landmarkerPromise: Promise<FaceLandmarker> | null = null

// Load-progress broadcast so the UI can show the one-time model download.
// frac < 0 means indeterminate (runtime/wasm phase); 0..1 is the model download.
const channel = progressChannel()
export const onLandmarkerProgress = channel.on

async function getLandmarker(): Promise<FaceLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      try {
        channel.emit({ loading: true, frac: -1 }) // preparing runtime (wasm)
        const fileset = await FilesetResolver.forVisionTasks(MODELS.mediapipeWasm)
        const buf = await fetchWithProgress(MODELS.landmarkerTask, (f) =>
          channel.emit({ loading: true, frac: f }),
        )
        const lm = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetBuffer: new Uint8Array(buf), delegate: 'CPU' },
          runningMode: 'IMAGE',
          numFaces: 4,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        })
        channel.emit({ loading: false, frac: 1 })
        return lm
      } catch (e) {
        channel.emit({ loading: false, frac: 1 })
        throw e
      }
    })()
  }
  return landmarkerPromise
}

export async function detectLandmarks(bitmap: ImageBitmap): Promise<Landmarks | null> {
  const lm = await getLandmarker()
  const w = bitmap.width
  const h = bitmap.height
  const res = lm.detect(bitmap)
  if (!res.faceLandmarks || res.faceLandmarks.length === 0) return null

  // Largest face by bounding-box area.
  let best = res.faceLandmarks[0]
  let bestArea = -1
  for (const face of res.faceLandmarks) {
    let minX = 1,
      minY = 1,
      maxX = 0,
      maxY = 0
    for (const p of face) {
      if (p.x < minX) minX = p.x
      if (p.y < minY) minY = p.y
      if (p.x > maxX) maxX = p.x
      if (p.y > maxY) maxY = p.y
    }
    const area = (maxX - minX) * (maxY - minY)
    if (area > bestArea) {
      bestArea = area
      best = face
    }
  }

  const points = new Float32Array(best.length * 2)
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity
  for (let i = 0; i < best.length; i++) {
    const x = best[i].x * w
    const y = best[i].y * h
    points[i * 2] = x
    points[i * 2 + 1] = y
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
  }
  return { points, box: { x: minX, y: minY, width: maxX - minX, height: maxY - minY } }
}
