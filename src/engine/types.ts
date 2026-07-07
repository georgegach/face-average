export const N_LANDMARKS = 478

// MediaPipe canonical indices we rely on.
export const IDX = {
  rightIris: 468,
  leftIris: 473,
  // A stable subset of semantically meaningful points for the editor.
  noseTip: 4,
  chin: 152,
  leftFace: 234,
  rightFace: 454,
} as const

export interface Landmarks {
  // Pixel coordinates in the source image, length N_LANDMARKS * 2 (x,y interleaved).
  points: Float32Array
  // Detection bounding box in source pixels.
  box: { x: number; y: number; width: number; height: number }
}

export interface Face {
  id: string
  name: string
  bitmap: ImageBitmap
  width: number
  height: number
  landmarks: Landmarks | null
  detecting: boolean
  failed: boolean
  weight: number
  enabled: boolean
  editRev: number // bumps when landmarks are manually edited
}

export interface AverageSettings {
  outWidth: number
  outHeight: number
  eyeDistance: number // fraction of width between iris centers
  eyeRatioY: number // eye line at height/eyeRatioY
  colorNormalize: boolean
  maskCompositing: boolean
  background: 'blur' | 'studio' | 'transparent' | 'none'
  templateId: string | null // face id whose geometry defines the shape, or null for average
}

export const DEFAULT_SETTINGS: AverageSettings = {
  outWidth: 768,
  outHeight: 1024,
  eyeDistance: 0.32,
  eyeRatioY: 2.6,
  colorNormalize: true,
  maskCompositing: false,
  background: 'blur',
  templateId: null,
}

export interface BabySettings {
  parentLean: number // -1 = all Parent A … 0 = 50/50 … +1 = all Parent B
  childAge: number // target apparent age of the child (slider ~5..12)
  deAge: boolean // apply FRAN de-aging (auto-skipped if the model is absent)
  outWidth: number
  outHeight: number
}

export const DEFAULT_BABY_SETTINGS: BabySettings = {
  parentLean: 0,
  childAge: 7,
  deAge: true,
  outWidth: 1024,
  outHeight: 1024,
}

export interface ReplaceSettings {
  feather: number // blend-ramp width as a fraction of the target face bbox width
  grow: number // face-oval expansion from its centroid
  colorMatch: number // Reinhard transfer strength 0..1
  blendTopK: 1 | 2 // number of nearest-pose sources blended (1 = max fidelity)
}

export const DEFAULT_REPLACE_SETTINGS: ReplaceSettings = {
  feather: 0.05,
  grow: 1.05,
  colorMatch: 0.7,
  blendTopK: 1,
}

export interface EditSettings {
  // Retouch
  skinSmooth: number // 0..1
  teethWhiten: number // 0..1
  // Makeup
  lipColor: string | null // hex, null = off
  lipStrength: number // 0..1
  blush: number // 0..1
  browDefine: number // 0..1
  eyeColor: string | null // hex, null = off
  eyeStrength: number // 0..1
  // Hair
  hairColor: string | null // hex, null = off
  hairStrength: number // 0..1
  // Scene
  background: 'none' | 'bokeh' | 'studio'
  backgroundStrength: number // 0..1
  vignette: number // 0..1
  // Shape (landmark-displacement warps)
  smile: number // -1..1
  eyeSize: number // -1..1
  noseSlim: number // 0..1
  faceSlim: number // 0..1
  hairVolume: number // 0..1
  // Age (FRAN re-aging; requires the optional CI-converted model)
  ageEnabled: boolean
  sourceAge: number // the person's actual age in the photo
  targetAge: number // desired apparent age
}

export const DEFAULT_EDIT_SETTINGS: EditSettings = {
  skinSmooth: 0,
  teethWhiten: 0,
  lipColor: null,
  lipStrength: 0.6,
  blush: 0,
  browDefine: 0,
  eyeColor: null,
  eyeStrength: 0.6,
  hairColor: null,
  hairStrength: 0.7,
  background: 'none',
  backgroundStrength: 0.6,
  vignette: 0,
  smile: 0,
  eyeSize: 0,
  noseSlim: 0,
  faceSlim: 0,
  hairVolume: 0,
  ageEnabled: false,
  sourceAge: 30,
  targetAge: 60,
}
