const MAX_DIM = 1600

export async function fileToBitmap(file: Blob): Promise<ImageBitmap> {
  const bmp = await createImageBitmap(file)
  return downscale(bmp)
}

export async function urlToBitmap(url: string): Promise<ImageBitmap> {
  const res = await fetch(url)
  const blob = await res.blob()
  return fileToBitmap(blob)
}

async function downscale(bmp: ImageBitmap): Promise<ImageBitmap> {
  const { width, height } = bmp
  const scale = Math.min(1, MAX_DIM / Math.max(width, height))
  if (scale >= 1) return bmp
  const w = Math.round(width * scale)
  const h = Math.round(height * scale)
  const resized = await createImageBitmap(bmp, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' })
  bmp.close()
  return resized
}

export function bitmapToDataURL(bmp: ImageBitmap, size = 96): string {
  const c = document.createElement('canvas')
  const scale = size / Math.max(bmp.width, bmp.height)
  c.width = Math.round(bmp.width * scale)
  c.height = Math.round(bmp.height * scale)
  c.getContext('2d')!.drawImage(bmp, 0, 0, c.width, c.height)
  return c.toDataURL('image/jpeg', 0.8)
}
