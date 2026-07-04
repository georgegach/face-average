/// <reference lib="webworker" />
import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision'

let landmarker: FaceLandmarker | null = null

async function ensure(wasmPath: string, modelPath: string) {
  if (landmarker) return landmarker
  const fileset = await FilesetResolver.forVisionTasks(wasmPath)
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    // CPU delegate: reliable in headless/worker contexts where WebGL may be
    // unavailable; detection is fast enough for interactive use.
    baseOptions: { modelAssetPath: modelPath, delegate: 'CPU' },
    runningMode: 'IMAGE',
    numFaces: 4,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  })
  return landmarker
}

interface DetectMsg {
  id: number
  type: 'detect'
  bitmap: ImageBitmap
  wasmPath: string
  modelPath: string
}

self.onmessage = async (e: MessageEvent<DetectMsg>) => {
  const msg = e.data
  if (msg.type !== 'detect') return
  try {
    const lm = await ensure(msg.wasmPath, msg.modelPath)
    const w = msg.bitmap.width
    const h = msg.bitmap.height
    const res = lm.detect(msg.bitmap)
    if (!res.faceLandmarks || res.faceLandmarks.length === 0) {
      ;(self as unknown as Worker).postMessage({ id: msg.id, ok: false })
      return
    }
    // Pick the largest face by bbox area.
    let best = 0
    let bestArea = -1
    res.faceLandmarks.forEach((face, i) => {
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
        best = i
      }
    })
    const face = res.faceLandmarks[best]
    const points = new Float32Array(face.length * 2)
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity
    for (let i = 0; i < face.length; i++) {
      const x = face[i].x * w
      const y = face[i].y * h
      points[i * 2] = x
      points[i * 2 + 1] = y
      if (x < minX) minX = x
      if (y < minY) minY = y
      if (x > maxX) maxX = x
      if (y > maxY) maxY = y
    }
    ;(self as unknown as Worker).postMessage(
      {
        id: msg.id,
        ok: true,
        points,
        box: { x: minX, y: minY, width: maxX - minX, height: maxY - minY },
      },
      { transfer: [points.buffer as ArrayBuffer] },
    )
  } catch (err) {
    ;(self as unknown as Worker).postMessage({
      id: msg.id,
      ok: false,
      error: String(err),
    })
  }
}
