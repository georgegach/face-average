import { useEffect, useRef, useState } from 'react'
import { useStore } from '../state/store'
import { Slider } from './Slider'
import { Icon } from './Icon'
import { exportPng, downloadBlob } from '../engine/export'
import { upscale, type UpscaleStage } from '../engine/upscale'
import type { UpscalerKind } from '../engine/models'

export function Controls() {
  const mode = useStore((s) => s.mode)
  if (mode === 'average') return <AveragePanel />
  if (mode === 'morph') return <MorphPanel />
  if (mode === 'replace') return <ReplacePanel />
  return <EnhancePanel />
}

function OutputSize() {
  const settings = useStore((s) => s.settings)
  const update = useStore((s) => s.updateSettings)
  const sizes: [string, number, number][] = [
    ['Portrait', 768, 1024],
    ['Square', 1024, 1024],
    ['Large', 1024, 1365],
  ]
  return (
    <div>
      <div className="label mb-1">Output size</div>
      <div className="flex gap-2">
        {sizes.map(([name, w, h]) => (
          <button
            key={name}
            onClick={() => update({ outWidth: w, outHeight: h })}
            className={`btn text-xs flex-1 ${
              settings.outWidth === w && settings.outHeight === h ? 'btn-accent' : 'btn-ghost'
            }`}
          >
            {name}
          </button>
        ))}
      </div>
    </div>
  )
}

function AveragePanel() {
  const settings = useStore((s) => s.settings)
  const update = useStore((s) => s.updateSettings)
  const runAverage = useStore((s) => s.runAverage)
  const computing = useStore((s) => s.computing)
  const result = useStore((s) => s.result)
  const faces = useStore((s) => s.faces)
  const usable = faces.filter((f) => f.enabled && f.landmarks && !f.failed).length
  const hasResult = !!result
  const [adv, setAdv] = useState(false)

  // Re-render the average automatically when any setting or per-face weight
  // changes, but only after the user has produced a first result.
  const sig = JSON.stringify([
    settings.background,
    settings.colorNormalize,
    settings.templateId,
    settings.outWidth,
    settings.outHeight,
    settings.eyeDistance.toFixed(3),
    settings.eyeRatioY.toFixed(3),
    faces.map((f) => `${f.id}:${f.weight}:${f.enabled}:${!!f.landmarks}:${f.editRev}`),
  ])
  useEffect(() => {
    if (!hasResult) return
    const t = setTimeout(() => runAverage(), 140)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig])

  const doExport = async (scale: number) => {
    if (!result) return
    downloadBlob(await exportPng(result, scale), `facestudio-average${scale > 1 ? '@2x' : ''}.png`)
  }

  return (
    <div className="flex flex-col gap-4">
      <button className="btn-accent" disabled={usable < 1 || computing} onClick={runAverage}>
        {computing ? 'Averaging…' : `Average ${usable} face${usable === 1 ? '' : 's'}`}
      </button>

      <OutputSize />

      <div>
        <div className="label mb-1">Background</div>
        <div className="grid grid-cols-2 gap-2">
          {(['blur', 'studio', 'transparent', 'none'] as const).map((b) => (
            <button
              key={b}
              onClick={() => update({ background: b })}
              className={`btn text-xs capitalize ${
                settings.background === b ? 'btn-accent' : 'btn-ghost'
              }`}
            >
              {b}
            </button>
          ))}
        </div>
      </div>

      <label className="flex items-center justify-between text-sm">
        <span className="text-content">Colour normalise</span>
        <input
          type="checkbox"
          checked={settings.colorNormalize}
          onChange={(e) => update({ colorNormalize: e.target.checked })}
        />
      </label>

      {settings.templateId && (
        <div className="text-xs text-accent-hi">
          Using a template face for shape.{' '}
          <button className="underline" onClick={() => update({ templateId: null })}>
            reset
          </button>
        </div>
      )}

      <button className="text-xs text-muted hover:text-content text-left" onClick={() => setAdv(!adv)}>
        {adv ? '▾' : '▸'} Advanced alignment
      </button>
      {adv && (
        <div className="flex flex-col gap-3 panel p-3">
          <Slider
            label="Eye distance"
            value={settings.eyeDistance}
            min={0.2}
            max={0.45}
            step={0.01}
            onChange={(v) => update({ eyeDistance: v })}
            format={(v) => v.toFixed(2)}
          />
          <Slider
            label="Eye height"
            value={settings.eyeRatioY}
            min={2}
            max={3.5}
            step={0.05}
            onChange={(v) => update({ eyeRatioY: v })}
            format={(v) => v.toFixed(2)}
          />
        </div>
      )}

      {result && (
        <div className="flex gap-2 border-t border-edge/70 pt-3">
          <button className="btn-ghost flex-1 text-xs" onClick={() => doExport(1)}>
            PNG
          </button>
          <button className="btn-ghost flex-1 text-xs" onClick={() => doExport(2)}>
            PNG @2×
          </button>
        </div>
      )}
    </div>
  )
}

function MorphPanel() {
  // Select the stable array, then filter in render — returning a fresh array
  // from the selector would loop useSyncExternalStore (React #185).
  const faces = useStore((s) => s.faces).filter((f) => f.landmarks && !f.failed)
  const morphA = useStore((s) => s.morphA)
  const morphB = useStore((s) => s.morphB)
  const setPair = useStore((s) => s.setMorphPair)

  const picker = (label: string, value: string | null, set: (id: string) => void) => (
    <div>
      <div className="label mb-1">{label}</div>
      <select
        value={value ?? ''}
        onChange={(e) => set(e.target.value)}
        className="w-full bg-surface2 rounded-xl px-3 py-2 text-sm"
      >
        <option value="" disabled>
          choose…
        </option>
        {faces.map((f) => (
          <option key={f.id} value={f.id}>
            {f.name}
          </option>
        ))}
      </select>
    </div>
  )

  return (
    <div className="flex flex-col gap-4">
      {faces.length < 2 && (
        <p className="text-xs text-muted">Add at least two detected faces to morph.</p>
      )}
      {picker('Face A', morphA, (id) => setPair(id, morphB))}
      {picker('Face B', morphB, (id) => setPair(morphA, id))}
      <OutputSize />
      <p className="text-xs text-muted">
        Use the blend slider under the canvas to scrub, or export an animated morph.
      </p>
    </div>
  )
}

function ReplacePanel() {
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
  useEffect(() => {
    if (!result) return
    const t = setTimeout(() => runReplace(), 140)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig])

  const doExport = async (scale: number) => {
    if (!result) return
    downloadBlob(await exportPng(result, scale), `facestudio-replace${scale > 1 ? '@2x' : ''}.png`)
  }

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
            className={`panel border-dashed cursor-pointer text-center py-6 px-3 transition-colors ${
              drag ? 'border-accent bg-accent/5' : ''
            }`}
          >
            <Icon name="faces" size={20} className="mx-auto text-muted mb-1" />
            <p className="text-sm text-content">Drop the photo to swap into</p>
            <p className="text-xs text-muted mt-1">the largest face in it gets replaced</p>
          </div>
        ) : (
          <div className="panel p-2 flex gap-2 items-start">
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

      <div className="flex flex-col gap-3 panel p-3">
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

      {result && (
        <div className="flex gap-2 border-t border-edge/70 pt-3">
          <button className="btn-ghost flex-1 text-xs" onClick={() => doExport(1)}>
            PNG
          </button>
          <button className="btn-ghost flex-1 text-xs" onClick={() => doExport(2)}>
            PNG @2×
          </button>
        </div>
      )}
    </div>
  )
}

function EnhancePanel() {
  const result = useStore((s) => s.result)
  const setResult = useStore((s) => s.setResult)
  const [kind, setKind] = useState<UpscalerKind>('photo')
  const [progress, setProgress] = useState<number | null>(null)
  const [stage, setStage] = useState<UpscaleStage>('download')
  const [msg, setMsg] = useState<string | null>(null)

  const run = async () => {
    if (!result) return
    setMsg(null)
    setStage('download')
    setProgress(0)
    try {
      const up = await upscale(result, kind, (st, f) => {
        setStage(st)
        setProgress(f)
      })
      setResult(up)
      setMsg('Upscaled 4×.')
    } catch (e) {
      setMsg('Upscale failed: ' + (e as Error).message)
    } finally {
      setProgress(null)
    }
  }

  const kinds: [UpscalerKind, string][] = [
    ['photo', 'Photo'],
    ['anime', 'Anime / Art'],
    ['general', 'General / Nature'],
  ]

  return (
    <div className="flex flex-col gap-4">
      {!result && <p className="text-xs text-muted">Render an average or morph first.</p>}
      <div>
        <div className="label mb-1">Model</div>
        <div className="flex flex-col gap-2">
          {kinds.map(([k, name]) => (
            <button
              key={k}
              onClick={() => setKind(k)}
              className={`btn text-xs ${kind === k ? 'btn-accent' : 'btn-ghost'}`}
            >
              {name}
            </button>
          ))}
        </div>
      </div>
      <button className="btn-accent" disabled={!result || progress !== null} onClick={run}>
        {progress !== null
          ? `${stage === 'download' ? 'Downloading model' : 'Upscaling'} ${Math.round(progress * 100)}%`
          : 'Upscale 4×'}
      </button>
      {progress !== null && (
        <div className="h-1.5 rounded-full bg-surface3 overflow-hidden">
          <div
            className="h-full bg-accent transition-[width] duration-150"
            style={{ width: `${Math.round(progress * 100)}%` }}
          />
        </div>
      )}
      {msg && <p className="text-xs text-muted">{msg}</p>}
      <p className="text-[11px] text-faint">
        Runs on-device via ONNX (WebGPU when available). Large images may take a while.
      </p>
    </div>
  )
}
