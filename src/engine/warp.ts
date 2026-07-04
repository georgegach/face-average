// GPU piecewise-affine warp. Each triangle of a shared mesh is drawn at its
// destination position while sampling the source texture at the corresponding
// source position — producing a smooth per-triangle affine warp in one pass.

const VERT = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;   // destination, clip space
layout(location=1) in vec2 aUV;    // source, 0..1
out vec2 vUV;
void main() {
  vUV = aUV;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`

const FRAG = `#version 300 es
precision highp float;
uniform sampler2D uTex;
in vec2 vUV;
out vec4 outColor;
void main() {
  outColor = texture(uTex, vUV);
}`

function compile(gl: WebGL2RenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!
  gl.shaderSource(s, src)
  gl.compileShader(s)
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error('shader: ' + gl.getShaderInfoLog(s))
  }
  return s
}

export class WarpEngine {
  private gl: WebGL2RenderingContext
  private prog: WebGLProgram
  private vbo: WebGLBuffer
  private tex: WebGLTexture
  readonly canvas: OffscreenCanvas | HTMLCanvasElement
  private w = 0
  private h = 0

  constructor() {
    this.canvas =
      typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(1, 1)
        : document.createElement('canvas')
    const gl = this.canvas.getContext('webgl2', {
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
    }) as WebGL2RenderingContext | null
    if (!gl) throw new Error('WebGL2 unavailable')
    this.gl = gl
    const prog = gl.createProgram()!
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT))
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG))
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('link: ' + gl.getProgramInfoLog(prog))
    }
    this.prog = prog
    this.vbo = gl.createBuffer()!
    this.tex = gl.createTexture()!
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
  }

  private resize(w: number, h: number) {
    if (this.w === w && this.h === h) return
    this.w = w
    this.h = h
    this.canvas.width = w
    this.canvas.height = h
    this.gl.viewport(0, 0, w, h)
  }

  /**
   * @param source  aligned source image (already similarity-transformed to w×h)
   * @param srcPts  source points in output pixels (interleaved), N points
   * @param dstPts  destination points in output pixels (interleaved), N points
   * @param tris    triangle vertex indices (into the N points), length 3*T
   */
  warp(
    source: ImageBitmap | HTMLCanvasElement | OffscreenCanvas,
    srcPts: Float32Array,
    dstPts: Float32Array,
    tris: Uint32Array,
    w: number,
    h: number,
  ): Uint8ClampedArray {
    const gl = this.gl
    this.resize(w, h)
    gl.useProgram(this.prog)

    // Build interleaved vertex buffer [posX,posY,u,v] per triangle vertex.
    const nVerts = tris.length
    const data = new Float32Array(nVerts * 4)
    for (let i = 0; i < nVerts; i++) {
      const idx = tris[i]
      const dx = dstPts[idx * 2]
      const dy = dstPts[idx * 2 + 1]
      const sx = srcPts[idx * 2]
      const sy = srcPts[idx * 2 + 1]
      data[i * 4] = (dx / w) * 2 - 1
      data[i * 4 + 1] = 1 - (dy / h) * 2
      data[i * 4 + 2] = sx / w
      data[i * 4 + 3] = 1 - sy / h // flipY compensation
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo)
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0)
    gl.enableVertexAttribArray(1)
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8)

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.tex)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source as TexImageSource)
    gl.uniform1i(gl.getUniformLocation(this.prog, 'uTex'), 0)

    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    gl.drawArrays(gl.TRIANGLES, 0, nVerts)

    const out = new Uint8ClampedArray(w * h * 4)
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out)
    // readPixels is bottom-up; flip to top-down to match ImageData.
    return flipRows(out, w, h)
  }
}

function flipRows(px: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(px.length)
  const rb = w * 4
  for (let y = 0; y < h; y++) {
    const src = y * rb
    const dst = (h - 1 - y) * rb
    out.set(px.subarray(src, src + rb), dst)
  }
  return out
}

let shared: WarpEngine | null = null
export function getWarpEngine(): WarpEngine {
  if (!shared) shared = new WarpEngine()
  return shared
}
