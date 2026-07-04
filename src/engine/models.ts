// Runtime paths to self-hosted model assets. BASE_URL is '/face-average/' in
// production and '/' in dev, so these resolve same-origin in both.
const B = import.meta.env.BASE_URL

export const MODELS = {
  landmarkerTask: `${B}models/face_landmarker.task`,
  mediapipeWasm: `${B}models/wasm`,
  faceParsing: `${B}models/face_parsing_bisenet.onnx`,
  upscalers: {
    photo: `${B}models/realesrgan-x4-photo.onnx`,
    anime: `${B}models/realesrgan-x4-anime.onnx`,
    general: `${B}models/realesr-general-x4v3.onnx`,
  },
} as const

export type UpscalerKind = keyof typeof MODELS.upscalers
