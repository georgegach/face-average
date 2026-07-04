import { useEffect, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent } from 'react'
import { useStore } from '../state/store'
import { Icon } from './Icon'
import { Slider } from './Slider'

// Manual landmark editor: grab any point and drag it; nearby points follow with
// a Gaussian falloff so the mesh deforms smoothly. Wheel zooms, background drag
// pans. Used to fix slightly-off detections before averaging/morphing.
export function LandmarkEditor({ faceId, onClose }: { faceId: string; onClose: () => void }) {
  const face = useStore((s) => s.faces.find((f) => f.id === faceId))
  const setLandmarks = useStore((s) => s.setLandmarks)

  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const pointsRef = useRef<Float32Array>(new Float32Array(0))
  const originalRef = useRef<Float32Array>(new Float32Array(0))
  const view = useRef({ scale: 1, tx: 0, ty: 0 })
  const drag = useRef<
    | null
    | { mode: 'pan'; sx: number; sy: number; tx: number; ty: number }
    | { mode: 'point'; index: number; snap: Float32Array; startX: number; startY: number }
  >(null)
  const hoverRef = useRef<number>(-1)

  const [radius, setRadius] = useState(30)
  const [dirty, setDirty] = useState(false)

  // Initialise working copy from the face's landmarks.
  useEffect(() => {
    if (!face?.landmarks) return
    pointsRef.current = face.landmarks.points.slice()
    originalRef.current = face.landmarks.points.slice()
    setRadius(Math.max(12, Math.round(face.landmarks.box.width * 0.12)))
    fitView()
    draw()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [faceId])

  function fitView() {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap || !face) return
    const cw = wrap.clientWidth
    const ch = wrap.clientHeight
    canvas.width = cw
    canvas.height = ch
    const s = Math.min(cw / face.width, ch / face.height) * 0.92
    view.current = {
      scale: s,
      tx: (cw - face.width * s) / 2,
      ty: (ch - face.height * s) / 2,
    }
  }

  function toScreen(x: number, y: number): [number, number] {
    const v = view.current
    return [x * v.scale + v.tx, y * v.scale + v.ty]
  }
  function toImg(sx: number, sy: number): [number, number] {
    const v = view.current
    return [(sx - v.tx) / v.scale, (sy - v.ty) / v.scale]
  }

  function draw() {
    const canvas = canvasRef.current
    if (!canvas || !face) return
    const ctx = canvas.getContext('2d')!
    const v = view.current
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.setTransform(v.scale, 0, 0, v.scale, v.tx, v.ty)
    ctx.drawImage(face.bitmap, 0, 0)
    ctx.setTransform(1, 0, 0, 1, 0, 0)

    const pts = pointsRef.current
    const r = Math.max(1.5, v.scale * 1.4)
    ctx.fillStyle = 'rgba(34, 211, 238, 0.85)'
    for (let i = 0; i < pts.length; i += 2) {
      const [sx, sy] = toScreen(pts[i], pts[i + 1])
      ctx.beginPath()
      ctx.arc(sx, sy, r, 0, Math.PI * 2)
      ctx.fill()
    }
    // Highlight hovered/active point.
    const dc = drag.current
    const hi = dc?.mode === 'point' ? dc.index : hoverRef.current
    if (hi >= 0) {
      const [sx, sy] = toScreen(pts[hi * 2], pts[hi * 2 + 1])
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(sx, sy, r + 4, 0, Math.PI * 2)
      ctx.stroke()
    }
  }

  function nearestPoint(sx: number, sy: number, maxScreenDist: number): number {
    const pts = pointsRef.current
    let best = -1
    let bestD = maxScreenDist * maxScreenDist
    for (let i = 0; i < pts.length; i += 2) {
      const [px, py] = toScreen(pts[i], pts[i + 1])
      const d = (px - sx) * (px - sx) + (py - sy) * (py - sy)
      if (d < bestD) {
        bestD = d
        best = i / 2
      }
    }
    return best
  }

  function onPointerDown(e: ReactPointerEvent) {
    const rect = canvasRef.current!.getBoundingClientRect()
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top
    const idx = nearestPoint(sx, sy, 14)
    canvasRef.current!.setPointerCapture(e.pointerId)
    if (idx >= 0) {
      const [ix, iy] = toImg(sx, sy)
      drag.current = { mode: 'point', index: idx, snap: pointsRef.current.slice(), startX: ix, startY: iy }
    } else {
      drag.current = { mode: 'pan', sx, sy, tx: view.current.tx, ty: view.current.ty }
    }
  }

  function onPointerMove(e: ReactPointerEvent) {
    const rect = canvasRef.current!.getBoundingClientRect()
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top
    const d = drag.current
    if (!d) {
      const idx = nearestPoint(sx, sy, 14)
      if (idx !== hoverRef.current) {
        hoverRef.current = idx
        draw()
      }
      return
    }
    if (d.mode === 'pan') {
      view.current.tx = d.tx + (sx - d.sx)
      view.current.ty = d.ty + (sy - d.sy)
      draw()
      return
    }
    // point drag with Gaussian falloff
    const [ix, iy] = toImg(sx, sy)
    const dx = ix - d.startX
    const dy = iy - d.startY
    const pts = pointsRef.current
    const snap = d.snap
    const ax = snap[d.index * 2]
    const ay = snap[d.index * 2 + 1]
    const sigma2 = 2 * radius * radius
    const cutoff = (3 * radius) * (3 * radius)
    for (let i = 0; i < pts.length; i += 2) {
      const ddx = snap[i] - ax
      const ddy = snap[i + 1] - ay
      const dist2 = ddx * ddx + ddy * ddy
      if (dist2 > cutoff) continue
      const w = Math.exp(-dist2 / sigma2)
      pts[i] = snap[i] + w * dx
      pts[i + 1] = snap[i + 1] + w * dy
    }
    draw()
  }

  function onPointerUp() {
    if (drag.current?.mode === 'point') setDirty(true)
    drag.current = null
  }

  function onWheel(e: ReactWheelEvent) {
    const rect = canvasRef.current!.getBoundingClientRect()
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top
    const [ix, iy] = toImg(sx, sy)
    const factor = Math.exp(-e.deltaY * 0.0015)
    const v = view.current
    v.scale = Math.min(40, Math.max(0.05, v.scale * factor))
    v.tx = sx - ix * v.scale
    v.ty = sy - iy * v.scale
    draw()
  }

  function reset() {
    pointsRef.current = originalRef.current.slice()
    setDirty(false)
    draw()
  }

  function save() {
    setLandmarks(faceId, pointsRef.current.slice())
    onClose()
  }

  useEffect(() => {
    draw()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [radius])

  useEffect(() => {
    const onResize = () => {
      fitView()
      draw()
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (!face || !face.landmarks) return null

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex flex-col p-3 sm:p-6" onClick={onClose}>
      <div
        className="panel flex-1 min-h-0 flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-edge/70">
          <div>
            <h3 className="text-sm font-semibold">Edit landmarks — {face.name}</h3>
            <p className="text-[11px] text-muted">
              Drag any point to fix it; neighbours follow. Scroll to zoom, drag background to pan.
            </p>
          </div>
          <button className="text-muted hover:text-content" onClick={onClose} aria-label="Close">
            <Icon name="close" size={16} />
          </button>
        </div>

        <div ref={wrapRef} className="flex-1 min-h-0 bg-bg relative touch-none">
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full cursor-crosshair"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={() => {
              if (!drag.current && hoverRef.current !== -1) {
                hoverRef.current = -1
                draw()
              }
            }}
            onWheel={onWheel}
          />
        </div>

        <div className="flex items-center gap-4 px-4 py-3 border-t border-edge/70 flex-wrap">
          <div className="w-40">
            <Slider
              label="Falloff"
              value={radius}
              min={4}
              max={160}
              step={1}
              onChange={setRadius}
              format={(v) => `${v}px`}
            />
          </div>
          <div className="flex-1" />
          <button className="btn-ghost text-xs" onClick={reset} disabled={!dirty}>
            Reset
          </button>
          <button className="btn-ghost text-xs" onClick={onClose}>
            Cancel
          </button>
          <button className="btn-accent text-xs" onClick={save} disabled={!dirty}>
            Save
          </button>
        </div>
      </div>
    </div>
  )
}
