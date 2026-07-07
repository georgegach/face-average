import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../state/store'
import { drawFitted, drawFittedBitmap } from './canvasFit'
import { MorphSession, morphSchedule } from '../engine/morph'
import { exportVideo, exportGif, downloadBlob } from '../engine/export'
import { capture } from '../lib/analytics'
import { PresetGallery } from './PresetGallery'
import { Slider } from './Slider'
import { Icon } from './Icon'

export function Stage() {
  const mode = useStore((s) => s.mode)
  if (mode === 'morph') return <MorphStage />
  return <ResultStage />
}

/** Full-area overlay showing the current pipeline step with a determinate bar
 *  (or an indeterminate pulse when the step's duration is unknown). */
function ProgressOverlay({ fallback }: { fallback: string }) {
  const progress = useStore((s) => s.progress)
  return (
    <div className="absolute inset-0 grid place-items-center bg-bg/50 backdrop-blur-sm z-10">
      <div className="panel px-5 py-4 w-72 flex flex-col gap-2.5">
        <div className="text-sm font-medium text-content">{progress?.label ?? fallback}</div>
        <div className="h-1 rounded-full bg-surface3 overflow-hidden">
          {progress && progress.frac >= 0 ? (
            <div
              className="h-full bg-accent transition-[width] duration-150 rounded-full"
              style={{ width: `${Math.round(Math.min(1, progress.frac) * 100)}%` }}
            />
          ) : (
            <div className="h-full w-1/3 bg-accent animate-pulse rounded-full" />
          )}
        </div>
      </div>
    </div>
  )
}

function ResultStage() {
  const result = useStore((s) => s.result)
  const computing = useStore((s) => s.computing)
  const error = useStore((s) => s.error)
  const faces = useStore((s) => s.faces)
  const mode = useStore((s) => s.mode)
  const editFaceId = useStore((s) => s.editFaceId)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [comparing, setComparing] = useState(false)

  // Before/after compare only makes sense in the editor, against the selected face.
  const editFace = mode === 'edit' ? faces.find((f) => f.id === editFaceId) : undefined
  const canCompare = !!result && !!editFace

  useEffect(() => {
    const c = canvasRef.current
    if (!c) return
    if (comparing && editFace) drawFittedBitmap(c, editFace.bitmap)
    else if (result) drawFitted(c, result)
  }, [result, comparing, editFace])

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
        <ProgressOverlay
          fallback={
            mode === 'replace'
              ? 'Replacing…'
              : mode === 'edit'
                ? 'Applying edits…'
                : mode === 'baby'
                  ? 'Generating…'
                  : 'Averaging…'
          }
        />
      )}
      {error && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 panel px-4 py-2 text-sm text-red-400 z-10">
          {error}
        </div>
      )}
      {result ? (
        <>
          <canvas
            ref={canvasRef}
            data-testid="result-canvas"
            className="max-w-full max-h-full rounded-2xl shadow-glass object-contain"
          />
          {canCompare && (
            <button
              className="absolute bottom-4 left-1/2 -translate-x-1/2 btn-ghost text-xs z-10 select-none touch-none"
              onPointerDown={() => setComparing(true)}
              onPointerUp={() => setComparing(false)}
              onPointerLeave={() => setComparing(false)}
              onPointerCancel={() => setComparing(false)}
            >
              <Icon name={comparing ? 'photo' : 'wand'} size={13} />
              {comparing ? 'Original' : 'Hold to compare'}
            </button>
          )}
        </>
      ) : mode === 'replace' ? (
        <div className="card px-5 py-4 text-muted text-sm text-center max-w-xs">
          <Icon name="swap" size={22} className="mx-auto mb-2 text-accent" />
          Drop a target photo in the right panel — the face in it gets replaced with your
          sources.
        </div>
      ) : mode === 'edit' ? (
        <div className="card px-5 py-4 text-muted text-sm text-center max-w-xs">
          <Icon name="wand" size={22} className="mx-auto mb-2 text-accent" />
          Pick a face and adjust the tools on the right, then press{' '}
          <span className="text-accent-hi font-medium">Apply edits</span>.
        </div>
      ) : mode === 'baby' ? (
        <div className="card px-5 py-4 text-muted text-sm text-center max-w-xs">
          <Icon name="heart" size={22} className="mx-auto mb-2 text-accent" />
          Pick two parents on the right, then press{' '}
          <span className="text-accent-hi font-medium">Generate baby</span>.
        </div>
      ) : (
        <div className="card px-5 py-4 text-muted text-sm text-center max-w-xs">
          <Icon name="faces" size={22} className="mx-auto mb-2 text-accent" />
          {faces.length} face{faces.length > 1 ? 's' : ''} loaded — press{' '}
          <span className="text-accent-hi font-medium">Average</span> to render.
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
  const setProgress = useStore((s) => s.setProgress)
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
    capture('export', { format: kind, tool: 'morph' })
    setBusy(true)
    try {
      const ts = morphSchedule(48, boomerang)
      const frames: ImageData[] = []
      for (let i = 0; i < ts.length; i++) {
        setProgress({ label: `Rendering frame ${i + 1}/${ts.length}`, frac: i / ts.length })
        frames.push(session.renderFrame(ts[i]))
        // Yield periodically so the progress bar paints during the render burst.
        if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0))
      }
      const fps = 24
      if (kind === 'gif') {
        const blob = await exportGif(frames, fps, (f) =>
          setProgress({ label: 'Encoding GIF', frac: f }),
        )
        downloadBlob(blob, 'facestudio-morph.gif')
      } else {
        const blob = await exportVideo(frames, fps, (f) =>
          setProgress({ label: 'Encoding video', frac: f }),
        )
        const ext = blob.type.includes('mp4') ? 'mp4' : 'webm'
        downloadBlob(blob, `facestudio-morph.${ext}`)
      }
    } finally {
      setBusy(false)
      setProgress(null)
    }
  }

  if (!faceA || !faceB) {
    return (
      <div className="flex-1 grid place-items-center p-6">
        <div className="card px-5 py-4 text-muted text-sm text-center max-w-xs">
          <Icon name="blend" size={22} className="mx-auto mb-2 text-accent" />
          <p className="text-content font-medium">Pick two faces to morph</p>
          <p className="mt-1">
            Add at least two detected faces, then choose A and B in the right panel.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 p-6 relative">
      {busy && <ProgressOverlay fallback="Rendering…" />}
      <canvas
        ref={canvasRef}
        data-testid="morph-canvas"
        className="max-w-full max-h-[60vh] rounded-2xl shadow-glass object-contain"
      />
      <div className="w-full max-w-md panel p-4 flex flex-col gap-3">
        <div className="flex justify-between text-xs text-muted">
          <span>{faceA.name}</span>
          <span>{faceB.name}</span>
        </div>
        <Slider label="Blend" value={t} min={0} max={1} step={0.01} onChange={setT} format={(v) => `${Math.round(v * 100)}%`} />
        <label className="flex items-center gap-2 text-xs text-muted">
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
