import { create } from 'zustand'
import {
  DEFAULT_SETTINGS,
  DEFAULT_REPLACE_SETTINGS,
  DEFAULT_EDIT_SETTINGS,
  type AverageSettings,
  type ReplaceSettings,
  type EditSettings,
  type Face,
} from '../engine/types'
import { detectLandmarks, onLandmarkerProgress, type LoadState } from '../engine/landmarks'
import { fileToBitmap, urlToBitmap, bitmapToDataURL } from '../engine/image'
import { computeAverage } from '../engine/average'
import { computeReplace } from '../engine/replace'
import { computeEdit } from '../engine/edit'
import { getParsing, onParsingProgress } from '../engine/parsing'
import { computeAge, onAgingProgress } from '../engine/aging'

export type Mode = 'average' | 'morph' | 'enhance' | 'replace' | 'edit'

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
  target: FaceView | null
  replaceSettings: ReplaceSettings
  replaceInfo: string | null
  editFaceId: string | null
  editSettings: EditSettings
  parseLoad: LoadState
  ageLoad: LoadState
  ageProgress: number | null
  // Global step-level progress for any running pipeline; frac -1 = indeterminate.
  progress: { label: string; frac: number } | null
  // Batch ingest/detection progress for the face tray.
  detectQueue: { done: number; total: number } | null

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
  setTargetFromFiles: (files: FileList | File[]) => Promise<void>
  clearTarget: () => void
  updateReplaceSettings: (patch: Partial<ReplaceSettings>) => void
  runReplace: () => void
  setEditFace: (id: string | null) => void
  updateEditSettings: (patch: Partial<EditSettings>) => void
  resetEditSettings: () => void
  runEdit: () => void
  setProgress: (p: { label: string; frac: number } | null) => void
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
  target: null,
  replaceSettings: { ...DEFAULT_REPLACE_SETTINGS },
  replaceInfo: null,
  editFaceId: null,
  editSettings: { ...DEFAULT_EDIT_SETTINGS },
  parseLoad: { loading: false, frac: 1 },
  ageLoad: { loading: false, frac: 1 },
  ageProgress: null,
  progress: null,
  detectQueue: null,

  setMode: (m) => set({ mode: m }),

  setProgress: (p) => set({ progress: p }),

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
    if (arr.length === 0) return
    set({ detectQueue: { done: 0, total: arr.length } })
    for (const file of arr) {
      try {
        const bmp = await fileToBitmap(file)
        await get().addBitmap(bmp, file.name.replace(/\.[^.]+$/, ''))
      } catch {
        /* skip unreadable file */
      }
      set((s) => (s.detectQueue ? { detectQueue: { ...s.detectQueue, done: s.detectQueue.done + 1 } } : {}))
    }
    set({ detectQueue: null })
  },

  addFromUrls: async (urls, names) => {
    if (urls.length === 0) return
    set({ detectQueue: { done: 0, total: urls.length } })
    for (let i = 0; i < urls.length; i++) {
      try {
        const bmp = await urlToBitmap(urls[i])
        const nm = names?.[i] ?? urls[i].split('/').pop()?.replace(/\.[^.]+$/, '') ?? 'face'
        await get().addBitmap(bmp, nm)
      } catch {
        /* skip */
      }
      set((s) => (s.detectQueue ? { detectQueue: { ...s.detectQueue, done: s.detectQueue.done + 1 } } : {}))
    }
    set({ detectQueue: null })
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
    set({ computing: true, error: null, progress: { label: 'Preparing faces', frac: 0 } })
    // Defer so the overlay paints before the heavy work starts.
    setTimeout(async () => {
      try {
        const res = await computeAverage(faces, settings, (label, frac) =>
          set({ progress: { label, frac } }),
        )
        set({ result: res.imageData, computing: false, progress: null })
      } catch (e) {
        set({ error: (e as Error).message, computing: false, progress: null })
      }
    }, 30)
  },

  setTargetFromFiles: async (files) => {
    const file = Array.from(files).find((f) => f.type.startsWith('image/'))
    if (!file) return
    let bmp: ImageBitmap
    try {
      bmp = await fileToBitmap(file)
    } catch {
      return // unreadable file
    }
    get().target?.bitmap.close() // free the previous target
    // Unique id (not a constant) so a stale detection callback from an old target can't
    // overwrite a newer one — see the guard below.
    const t: FaceView = {
      id: newId(),
      name: file.name.replace(/\.[^.]+$/, ''),
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
    set({ target: t })
    try {
      const lm = await detectLandmarks(bmp)
      set((s) => {
        const cur = s.target
        if (!cur || cur.id !== t.id) return {} // a newer target replaced this one mid-detect
        return { target: { ...cur, landmarks: lm, detecting: false, failed: !lm } }
      })
    } catch {
      set((s) => {
        const cur = s.target
        if (!cur || cur.id !== t.id) return {}
        return { target: { ...cur, detecting: false, failed: true } }
      })
    }
  },

  clearTarget: () =>
    set((s) => {
      s.target?.bitmap.close()
      return { target: null }
    }),

  updateReplaceSettings: (patch) =>
    set((s) => ({ replaceSettings: { ...s.replaceSettings, ...patch } })),

  runReplace: () => {
    const { faces, target, replaceSettings } = get()
    if (!target) {
      set({ error: 'Drop a target photo first' })
      return
    }
    set({ computing: true, error: null, progress: { label: 'Preparing replace', frac: 0 } })
    // Defer so the overlay paints before the heavy work starts.
    setTimeout(async () => {
      try {
        const res = await computeReplace(faces, target, replaceSettings, (label, frac) =>
          set({ progress: { label, frac } }),
        )
        set({
          result: res.imageData,
          replaceInfo: res.sourceName,
          computing: false,
          progress: null,
        })
      } catch (e) {
        set({ error: (e as Error).message, computing: false, progress: null })
      }
    }, 30)
  },

  setEditFace: (id) => set({ editFaceId: id }),

  updateEditSettings: (patch) =>
    set((s) => ({ editSettings: { ...s.editSettings, ...patch } })),

  resetEditSettings: () => set({ editSettings: { ...DEFAULT_EDIT_SETTINGS } }),

  runEdit: () => {
    const { faces, editFaceId, editSettings } = get()
    const face =
      faces.find((f) => f.id === editFaceId && f.landmarks && !f.failed) ??
      faces.find((f) => f.landmarks && !f.failed)
    if (!face) {
      set({ error: 'Add a face with a detected face first' })
      return
    }
    set({
      computing: true,
      error: null,
      editFaceId: face.id,
      progress: { label: 'Analysing face regions', frac: -1 },
    })
    // Defer so the overlay paints; parsing/aging may also download models on first use.
    setTimeout(async () => {
      try {
        const parsing = await getParsing(face)
        let base: ImageData | undefined
        if (editSettings.ageEnabled) {
          set({ ageProgress: 0 })
          base = await computeAge(face, editSettings.sourceAge, editSettings.targetAge, (f) =>
            set({ ageProgress: f, progress: { label: 'Re-aging face', frac: f } }),
          )
          set({ ageProgress: null })
        }
        const res = await computeEdit(face, parsing, editSettings, base, (label, frac) =>
          set({ progress: { label, frac } }),
        )
        set({ result: res, computing: false, progress: null })
      } catch (e) {
        set({
          error: (e as Error).message,
          computing: false,
          ageProgress: null,
          progress: null,
        })
      }
    }, 30)
  },
}))

// Mirror one-time landmarker model-download progress into the store.
onLandmarkerProgress((s) => useStore.setState({ modelLoad: s }))

// Mirror one-time face-parsing model-download progress into the store.
onParsingProgress((s) => useStore.setState({ parseLoad: s }))

// Mirror one-time re-aging model-download progress into the store.
onAgingProgress((s) => useStore.setState({ ageLoad: s }))
