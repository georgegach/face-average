import { GIFEncoder, quantize, applyPalette } from 'gifenc'

export function imageDataToCanvas(img: ImageData): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = img.width
  c.height = img.height
  c.getContext('2d')!.putImageData(img, 0, 0)
  return c
}

export async function exportPng(img: ImageData, scale = 1): Promise<Blob> {
  const base = imageDataToCanvas(img)
  let canvas: HTMLCanvasElement = base
  if (scale !== 1) {
    canvas = document.createElement('canvas')
    canvas.width = img.width * scale
    canvas.height = img.height * scale
    const ctx = canvas.getContext('2d')!
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(base, 0, 0, canvas.width, canvas.height)
  }
  return await new Promise((res) => canvas.toBlob((b) => res(b!), 'image/png'))
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

/** Encode frames to an animated GIF (fully client-side, no server). */
export async function exportGif(
  frames: ImageData[],
  fps: number,
  onProgress?: (frac: number) => void,
): Promise<Blob> {
  const enc = GIFEncoder()
  const delay = Math.round(1000 / fps)
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i]
    const palette = quantize(f.data, 256)
    const index = applyPalette(f.data, palette)
    enc.writeFrame(index, f.width, f.height, { palette, delay })
    onProgress?.((i + 1) / frames.length)
    await new Promise((r) => setTimeout(r, 0)) // let the progress bar paint
  }
  enc.finish()
  return new Blob([enc.bytes()], { type: 'image/gif' })
}

function pickVideoMime(): string {
  const cands = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4']
  for (const c of cands) if (MediaRecorder.isTypeSupported(c)) return c
  return 'video/webm'
}

/** Encode frames to WebM (or MP4 on Safari) via a canvas capture stream. */
export async function exportVideo(
  frames: ImageData[],
  fps: number,
  onProgress?: (frac: number) => void,
): Promise<Blob> {
  const w = frames[0].width
  const h = frames[0].height
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  const stream = canvas.captureStream(0)
  const track = stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack
  const mime = pickVideoMime()
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8_000_000 })
  const chunks: BlobPart[] = []
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
  const done = new Promise<Blob>((res) => {
    rec.onstop = () => res(new Blob(chunks, { type: mime }))
  })
  rec.start()
  const frameMs = 1000 / fps
  for (let i = 0; i < frames.length; i++) {
    ctx.putImageData(frames[i], 0, 0)
    track.requestFrame()
    onProgress?.((i + 1) / frames.length)
    await new Promise((r) => setTimeout(r, frameMs))
  }
  rec.stop()
  return done
}
