// Draws ImageData into a canvas element scaled to fit its container box while
// preserving aspect ratio. Returns nothing; mutates the canvas.
export function drawFitted(canvas: HTMLCanvasElement, img: ImageData) {
  canvas.width = img.width
  canvas.height = img.height
  canvas.getContext('2d')!.putImageData(img, 0, 0)
}
