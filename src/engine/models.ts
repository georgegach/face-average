// Runtime paths to self-hosted model assets. BASE_URL is '/face-average/' in
// production and '/' in dev, so these resolve same-origin in both.
const B = import.meta.env.BASE_URL

// Upscaler ONNX models are large (35–72 MB) and would blow past GitHub Pages'
// deployment size, so they load at runtime from HuggingFace (CORS-enabled) and
// are cached by the service worker. Everything else is self-hosted.
const HF = 'https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main'

export const MODELS = {
  landmarkerTask: `${B}models/face_landmarker.task`,
  mediapipeWasm: `${B}models/wasm`,
  ortWasm: `${B}models/ort/`,
  upscalers: {
    photo: `${HF}/4x-UltraSharpV2_Lite.onnx`,
    anime: `${HF}/4x-AnimeSharp.onnx`,
    general: `${HF}/4x-ClearRealityV1.onnx`,
  },
} as const

export type UpscalerKind = keyof typeof MODELS.upscalers
