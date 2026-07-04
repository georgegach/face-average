import { create } from 'zustand'
import { DEFAULT_SETTINGS, type AverageSettings, type Face } from '../engine/types'
import { detectLandmarks, onLandmarkerProgress, type LoadState } from '../engine/landmarks'
import { fileToBitmap, urlToBitmap, bitmapToDataURL } from '../engine/image'
import { computeAverage } from '../engine/average'

export type Mode = 'average' | 'morph' | 'enhance'

let idc = 0
const newId = () => `f${++idc}`

export interface FaceView extends Face {
  thumb: string
}

interface StoreState {
  faces: FaceView[]
  mode: Mode
  settings: AverageSettings
  result: ImageData | null
  computing: boolean
  error: string | null
  morphA: string | null
  morphB: string | null
  modelLoad: LoadState

  setMode: (m: Mode) => void
  addFiles: (files: FileList | File[]) => Promise<void>
  addFromUrls: (urls: string[], names?: string[]) => Promise<void>
  addBitmap: (bmp: ImageBitmap, name: string) => Promise<void>
  removeFace: (id: string) => void
  clearFaces: () => void
  setWeight: (id: string, weight: number) => void
  toggleEnabled: (id: string) => void
  setTemplate: (id: string | null) => void
  setLandmarks: (id: string, points: Float32Array) => void
  updateSettings: (patch: Partial<AverageSettings>) => void
  runAverage: () => void
  setResult: (img: ImageData | null) => void
  setMorphPair: (a: string | null, b: string | null) => void
}

export const useStore = create<StoreState>((set, get) => ({
  faces: [],
  mode: 'average',
  settings: { ...DEFAULT_SETTINGS },
  result: null,
  computing: false,
  error: null,
  morphA: null,
  morphB: null,
  modelLoad: { loading: false, frac: 1 },

  setMode: (m) => set({ mode: m }),

  addBitmap: async (bmp, name) => {
    const face: FaceView = {
      id: newId(),
      name,
      bitmap: bmp,
      width: bmp.width,
      height: bmp.height,
      landmarks: null,
      detecting: true,
      failed: false,
      weight: 1,
      enabled: true,
      editRev: 0,
      thumb: bitmapToDataURL(bmp),
    }
    set((s) => ({ faces: [...s.faces, face] }))
    try {
      const lm = await detectLandmarks(bmp)
      set((s) => ({
        faces: s.faces.map((f) =>
          f.id === face.id ? { ...f, landmarks: lm, detecting: false, failed: !lm } : f,
        ),
      }))
    } catch (err) {
      console.error('landmark detection failed', err)
      set((s) => ({
        faces: s.faces.map((f) => (f.id === face.id ? { ...f, detecting: false, failed: true } : f)),
      }))
    }
    // Seed morph selection with the first two faces.
    const { faces, morphA, morphB } = get()
    if (!morphA && faces[0]) set({ morphA: faces[0].id })
    else if (!morphB && faces[1]) set({ morphB: faces[1].id })
  },

  addFiles: async (files) => {
    const arr = Array.from(files).filter((f) => f.type.startsWith('image/'))
    for (const file of arr) {
      try {
        const bmp = await fileToBitmap(file)
        await get().addBitmap(bmp, file.name.replace(/\.[^.]+$/, ''))
      } catch {
        /* skip unreadable file */
      }
    }
  },

  addFromUrls: async (urls, names) => {
    for (let i = 0; i < urls.length; i++) {
      try {
        const bmp = await urlToBitmap(urls[i])
        const nm = names?.[i] ?? urls[i].split('/').pop()?.replace(/\.[^.]+$/, '') ?? 'face'
        await get().addBitmap(bmp, nm)
      } catch {
        /* skip */
      }
    }
  },

  removeFace: (id) =>
    set((s) => {
      const face = s.faces.find((f) => f.id === id)
      face?.bitmap.close()
      const settings =
        s.settings.templateId === id ? { ...s.settings, templateId: null } : s.settings
      return {
        faces: s.faces.filter((f) => f.id !== id),
        settings,
        morphA: s.morphA === id ? null : s.morphA,
        morphB: s.morphB === id ? null : s.morphB,
      }
    }),

  clearFaces: () =>
    set((s) => {
      s.faces.forEach((f) => f.bitmap.close())
      return { faces: [], result: null, morphA: null, morphB: null }
    }),

  setWeight: (id, weight) =>
    set((s) => ({ faces: s.faces.map((f) => (f.id === id ? { ...f, weight } : f)) })),

  toggleEnabled: (id) =>
    set((s) => ({ faces: s.faces.map((f) => (f.id === id ? { ...f, enabled: !f.enabled } : f)) })),

  setTemplate: (id) => set((s) => ({ settings: { ...s.settings, templateId: id } })),

  setLandmarks: (id, points) =>
    set((s) => ({
      faces: s.faces.map((f) => {
        if (f.id !== id || !f.landmarks) return f
        // Recompute bounding box from the edited points.
        let minX = Infinity,
          minY = Infinity,
          maxX = -Infinity,
          maxY = -Infinity
        for (let i = 0; i < points.length; i += 2) {
          const x = points[i]
          const y = points[i + 1]
          if (x < minX) minX = x
          if (y < minY) minY = y
          if (x > maxX) maxX = x
          if (y > maxY) maxY = y
        }
        return {
          ...f,
          landmarks: { points, box: { x: minX, y: minY, width: maxX - minX, height: maxY - minY } },
          editRev: f.editRev + 1,
        }
      }),
    })),

  updateSettings: (patch) => set((s) => ({ settings: { ...s.settings, ...patch } })),

  setResult: (img) => set({ result: img }),

  setMorphPair: (a, b) => set({ morphA: a, morphB: b }),

  runAverage: () => {
    const { faces, settings } = get()
    set({ computing: true, error: null })
    // Defer so the spinner paints before the synchronous heavy work.
    setTimeout(() => {
      try {
        const res = computeAverage(faces, settings)
        set({ result: res.imageData, computing: false })
      } catch (e) {
        set({ error: (e as Error).message, computing: false })
      }
    }, 30)
  },
}))

// Mirror one-time landmarker model-download progress into the store.
onLandmarkerProgress((s) => useStore.setState({ modelLoad: s }))
