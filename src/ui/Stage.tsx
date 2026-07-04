import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../state/store'
import { drawFitted } from './canvasFit'
import { MorphSession, morphSchedule } from '../engine/morph'
import { exportVideo, exportGif, downloadBlob } from '../engine/export'
import { PresetGallery } from './PresetGallery'
import { Slider } from './Slider'

export function Stage() {
  const mode = useStore((s) => s.mode)
  if (mode === 'morph') return <MorphStage />
  return <ResultStage />
}

function ResultStage() {
  const result = useStore((s) => s.result)
  const computing = useStore((s) => s.computing)
  const error = useStore((s) => s.error)
  const faces = useStore((s) => s.faces)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (result && canvasRef.current) drawFitted(canvasRef.current, result)
  }, [result])

  if (!result && faces.length === 0) {
    return (
      <div className="flex-1 grid place-items-center p-6">
        <PresetGallery />
      </div>
    )
  }

  return (
    <div className="flex-1 grid place-items-center p-6 relative">
      {computing && (
        <div className="absolute inset-0 grid place-items-center bg-ink-900/60 z-10">
          <div className="text-accent animate-pulse text-sm">Averaging…</div>
        </div>
      )}
      {error && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 panel px-4 py-2 text-sm text-red-400 z-10">
          {error}
        </div>
      )}
      {result ? (
        <canvas
          ref={canvasRef}
          data-testid="result-canvas"
          className="max-w-full max-h-full rounded-2xl shadow-glass object-contain"
        />
      ) : (
        <div className="text-slate-500 text-sm text-center">
          {faces.length} face{faces.length > 1 ? 's' : ''} loaded — press{' '}
          <span className="text-accent">Average</span> to render.
        </div>
      )}
    </div>
  )
}

function MorphStage() {
  const faces = useStore((s) => s.faces)
  const morphA = useStore((s) => s.morphA)
  const morphB = useStore((s) => s.morphB)
  const settings = useStore((s) => s.settings)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [t, setT] = useState(0.5)
  const [busy, setBusy] = useState(false)
  const [boomerang, setBoomerang] = useState(true)

  // Fall back to the first two detected faces if the selected pair is unset or
  // points at a face whose detection failed.
  const valid = faces.filter((f) => f.landmarks && !f.failed)
  const faceA = valid.find((f) => f.id === morphA) ?? valid[0]
  const faceB = valid.find((f) => f.id === morphB && f.id !== faceA?.id) ?? valid.find((f) => f.id !== faceA?.id)

  const session = useMemo(() => {
    if (!faceA || !faceB) return null
    try {
      return new MorphSession(faceA, faceB, settings)
    } catch {
      return null
    }
  }, [faceA, faceB, settings.outWidth, settings.outHeight])

  useEffect(() => {
    if (session && canvasRef.current) {
      drawFitted(canvasRef.current, session.renderFrame(t))
    }
  }, [session, t])

  const animate = async (kind: 'webm' | 'gif') => {
    if (!session) return
    setBusy(true)
    try {
      const ts = morphSchedule(48, boomerang)
      const frames = ts.map((tt) => session.renderFrame(tt))
      const fps = 24
      if (kind === 'gif') {
        const blob = exportGif(frames, fps)
        downloadBlob(blob, 'facestudio-morph.gif')
      } else {
        const blob = await exportVideo(frames, fps)
        const ext = blob.type.includes('mp4') ? 'mp4' : 'webm'
        downloadBlob(blob, `facestudio-morph.${ext}`)
      }
    } finally {
      setBusy(false)
    }
  }

  if (!faceA || !faceB) {
    return (
      <div className="flex-1 grid place-items-center p-6 text-center text-slate-400 text-sm">
        <div>
          <p>Pick two faces to morph.</p>
          <p className="text-slate-500 mt-1">
            Add at least two detected faces, then choose A and B in the right panel.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 p-6">
      <canvas
        ref={canvasRef}
        data-testid="morph-canvas"
        className="max-w-full max-h-[60vh] rounded-2xl shadow-glass object-contain"
      />
      <div className="w-full max-w-md panel p-4 flex flex-col gap-3">
        <div className="flex justify-between text-xs text-slate-400">
          <span>{faceA.name}</span>
          <span>{faceB.name}</span>
        </div>
        <Slider label="Blend" value={t} min={0} max={1} step={0.01} onChange={setT} format={(v) => `${Math.round(v * 100)}%`} />
        <label className="flex items-center gap-2 text-xs text-slate-400">
          <input type="checkbox" checked={boomerang} onChange={(e) => setBoomerang(e.target.checked)} />
          Boomerang
        </label>
        <div className="flex gap-2">
          <button className="btn-accent flex-1" disabled={busy} onClick={() => animate('webm')}>
            {busy ? 'Rendering…' : 'Export video'}
          </button>
          <button className="btn-ghost flex-1" disabled={busy} onClick={() => animate('gif')}>
            Export GIF
          </button>
        </div>
      </div>
    </div>
  )
}
