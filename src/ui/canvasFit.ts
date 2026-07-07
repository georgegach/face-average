// Draws ImageData into a canvas element scaled to fit its container box while
// preserving aspect ratio. Returns nothing; mutates the canvas.
export function drawFitted(canvas: HTMLCanvasElement, img: ImageData) {
  canvas.width = img.width
  canvas.height = img.height
  canvas.getContext('2d')!.putImageData(img, 0, 0)
}

// Draws a bitmap into a canvas at native resolution (CSS scales it to fit, like drawFitted).
export function drawFittedBitmap(canvas: HTMLCanvasElement, bmp: ImageBitmap) {
  canvas.width = bmp.width
  canvas.height = bmp.height
  canvas.getContext('2d')!.drawImage(bmp, 0, 0)
}
