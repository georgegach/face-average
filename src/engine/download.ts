// Fetch a binary asset while reporting download progress (0..1). Falls back to
// a plain arrayBuffer() when the stream or Content-Length is unavailable.
export async function fetchWithProgress(
  url: string,
  onProgress?: (frac: number) => void,
): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok && res.status !== 206) throw new Error(`download failed (${res.status})`)
  const total = Number(res.headers.get('content-length')) || 0
  if (!res.body || !total) {
    const buf = await res.arrayBuffer()
    onProgress?.(1)
    return buf
  }
  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.length
    onProgress?.(Math.min(1, received / total))
  }
  const out = new Uint8Array(received)
  let off = 0
  for (const c of chunks) {
    out.set(c, off)
    off += c.length
  }
  return out.buffer
}
