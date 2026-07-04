// Runtime paths to self-hosted model assets. BASE_URL is '/face-average/' in
// production and '/' in dev, so these resolve same-origin in both.
const B = import.meta.env.BASE_URL

export const MODELS = {
  landmarkerTask: `${B}models/face_landmarker.task`,
  mediapipeWasm: `${B}models/wasm`,
  ortWasm: `${B}models/ort/`,
  upscalers: {
    photo: `${B}models/upscale-photo.onnx`,
    anime: `${B}models/upscale-anime.onnx`,
    general: `${B}models/upscale-general.onnx`,
  },
} as const

export type UpscalerKind = keyof typeof MODELS.upscalers
