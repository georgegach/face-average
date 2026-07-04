import { MODELS } from './models'
import type { Landmarks } from './types'

// Main-thread client for the landmark worker. Keeps a single worker alive and
// serialises requests via incrementing ids.
let worker: Worker | null = null
let nextId = 1
const pending = new Map<
  number,
  { resolve: (l: Landmarks | null) => void; reject: (e: unknown) => void }
>()

function getWorker(): Worker {
  if (worker) return worker
  worker = new Worker(new URL('../workers/landmarks.worker.ts', import.meta.url), {
    type: 'module',
  })
  worker.onmessage = (e: MessageEvent) => {
    const { id, ok, points, box, error } = e.data
    const p = pending.get(id)
    if (!p) return
    pending.delete(id)
    if (error) p.reject(new Error(error))
    else if (!ok) p.resolve(null)
    else p.resolve({ points, box })
  }
  return worker
}

export async function detectLandmarks(bitmap: ImageBitmap): Promise<Landmarks | null> {
  const w = getWorker()
  const id = nextId++
  // The worker needs its own copy of the bitmap; clone via createImageBitmap.
  const clone = await createImageBitmap(bitmap)
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    w.postMessage(
      {
        id,
        type: 'detect',
        bitmap: clone,
        wasmPath: MODELS.mediapipeWasm,
        modelPath: MODELS.landmarkerTask,
      },
      [clone],
    )
  })
}
