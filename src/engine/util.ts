/** Yield to the event loop so progress UI can paint between heavy synchronous steps. */
export const yieldUI = () => new Promise<void>((r) => setTimeout(r, 0))

/** Progress callback: label + fraction (0..1), or frac -1 for indeterminate. */
export type OnProgress = (label: string, frac: number) => void

/** One-time model-load progress: frac 0..1, or -1 for indeterminate (wasm/runtime phase). */
export type LoadState = { loading: boolean; frac: number }

/**
 * A tiny listener registry for broadcasting one-time model-load progress to the
 * UI. Each lazily-loaded model (landmarker, face parser, re-aging net) owns one
 * channel: `on` subscribes (returns an unsubscribe), `emit` fans out.
 */
export function progressChannel() {
  const listeners = new Set<(s: LoadState) => void>()
  return {
    on(cb: (s: LoadState) => void): () => void {
      listeners.add(cb)
      return () => listeners.delete(cb)
    },
    emit(s: LoadState): void {
      for (const cb of listeners) cb(s)
    },
  }
}
