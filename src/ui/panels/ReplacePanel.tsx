import { useRef, useState } from 'react'
import { useStore } from '../../state/store'
import { Slider } from '../Slider'
import { Icon } from '../Icon'
import { ResultActions, useAutoRerun } from './shared'

export function ReplacePanel() {
  const faces = useStore((s) => s.faces)
  const target = useStore((s) => s.target)
  const rs = useStore((s) => s.replaceSettings)
  const update = useStore((s) => s.updateReplaceSettings)
  const setTargetFromFiles = useStore((s) => s.setTargetFromFiles)
  const clearTarget = useStore((s) => s.clearTarget)
  const runReplace = useStore((s) => s.runReplace)
  const computing = useStore((s) => s.computing)
  const result = useStore((s) => s.result)
  const replaceInfo = useStore((s) => s.replaceInfo)
  const inputRef = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)

  const usable = faces.filter((f) => f.enabled && f.landmarks && !f.failed).length
  const ready = usable >= 1 && !!target?.landmarks

  // Re-render automatically when any setting, the target, or the sources change, but only
  // after the user has produced a first result. Mirrors AveragePanel's debounce.
  const sig = JSON.stringify([
    rs.feather,
    rs.grow,
    rs.colorMatch,
    rs.blendTopK,
    target?.id,
    !!target?.landmarks,
    faces.map((f) => `${f.id}:${f.enabled}:${!!f.landmarks}:${f.editRev}`),
  ])
  useAutoRerun(sig, !!result, runReplace)

  return (
    <div className="flex flex-col gap-4">
      <button className="btn-accent" disabled={!ready || computing} onClick={runReplace}>
        {computing ? 'Replacing…' : 'Replace face'}
      </button>

      <p className="text-xs text-muted">
        {usable < 1
          ? 'Add source faces in the left tray — a few dozen angles of one person give the best match.'
          : `${usable} source face${usable === 1 ? '' : 's'} · the closest pose to the target is used.`}
      </p>

      <div>
        <div className="label mb-1">Target photo</div>
        {!target ? (
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setDrag(true)
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDrag(false)
              setTargetFromFiles(e.dataTransfer.files)
            }}
            className={`dropzone text-center py-6 px-3 ${drag ? 'border-accent bg-accent/5' : ''}`}
          >
            <Icon name="photo" size={20} className="mx-auto text-accent mb-1" />
            <p className="text-sm text-content">Drop the photo to swap into</p>
            <p className="text-xs text-muted mt-1">the largest face in it gets replaced</p>
          </div>
        ) : (
          <div className="card p-2 flex gap-2 items-start">
            <img src={target.thumb} alt={target.name} className="w-14 h-14 rounded-lg object-cover" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-content truncate flex-1">{target.name}</span>
                <button
                  title="Remove target"
                  aria-label="Remove target"
                  className="text-muted hover:text-red-400"
                  onClick={clearTarget}
                >
                  <Icon name="close" size={13} />
                </button>
              </div>
              <div className="text-[10px] mt-0.5">
                {target.detecting ? (
                  <span className="text-accent-hi">Detecting face…</span>
                ) : target.failed ? (
                  <span className="text-red-400">No face found — use a clear, front-facing photo</span>
                ) : (
                  <span className="text-emerald-400 inline-flex items-center gap-1">
                    <Icon name="check" size={11} /> Face detected
                  </span>
                )}
              </div>
            </div>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          data-testid="replace-target-input"
          onChange={(e) => e.target.files && setTargetFromFiles(e.target.files)}
        />
      </div>

      <div className="flex flex-col gap-3 card p-3">
        <Slider
          label="Colour match"
          value={rs.colorMatch}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => update({ colorMatch: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
        <Slider
          label="Face region"
          value={rs.grow}
          min={1.0}
          max={1.25}
          step={0.01}
          onChange={(v) => update({ grow: v })}
          format={(v) => v.toFixed(2)}
        />
        <Slider
          label="Edge feather"
          value={rs.feather}
          min={0.01}
          max={0.15}
          step={0.01}
          onChange={(v) => update({ feather: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
        <div>
          <div className="label mb-1">Blend poses</div>
          <div className="flex gap-2">
            {([1, 2] as const).map((k) => (
              <button
                key={k}
                onClick={() => update({ blendTopK: k })}
                className={`btn text-xs flex-1 ${rs.blendTopK === k ? 'btn-accent' : 'btn-ghost'}`}
              >
                {k === 1 ? 'Sharpest (1)' : 'Smoother (2)'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {replaceInfo && result && (
        <p className="text-[11px] text-faint">Source used: {replaceInfo}</p>
      )}

      {result && <ResultActions result={result} prefix="facestudio-replace" />}
    </div>
  )
}
