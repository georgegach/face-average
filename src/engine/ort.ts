// Shared ONNX Runtime Web bootstrap for the on-device models (face parsing,
// re-aging, upscalers). Centralises the identical setup every caller needs:
// self-hosted wasm assets, single-threaded (GitHub Pages isn't cross-origin
// isolated, so SharedArrayBuffer/threads are unavailable), and a WebGPU→wasm
// execution-provider preference.
import * as ort from 'onnxruntime-web'
import { MODELS } from './models'
import { fetchWithProgressCached } from './download'

/** Prefer the WebGPU execution provider when the browser exposes it, else wasm. */
function providers(): string[] {
  const p: string[] = []
  if ('gpu' in navigator) p.push('webgpu')
  p.push('wasm')
  return p
}

/**
 * Download (progress-reported, Cache-API-persisted) a model and create an ORT
 * session for it. Callers own their own memoisation and error-reset semantics;
 * this only covers the env setup + fetch + session creation they all share.
 */
export async function createOrtSession(
  url: string,
  onDownload?: (frac: number) => void,
): Promise<ort.InferenceSession> {
  ort.env.wasm.wasmPaths = MODELS.ortWasm
  ort.env.wasm.numThreads = 1
  const bytes = await fetchWithProgressCached(url, onDownload)
  return ort.InferenceSession.create(bytes, { executionProviders: providers() })
}
