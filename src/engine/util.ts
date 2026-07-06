/** Yield to the event loop so progress UI can paint between heavy synchronous steps. */
export const yieldUI = () => new Promise<void>((r) => setTimeout(r, 0))

/** Progress callback: label + fraction (0..1), or frac -1 for indeterminate. */
export type OnProgress = (label: string, frac: number) => void
