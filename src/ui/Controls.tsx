import { useEffect, useState } from 'react'
import { useStore } from '../state/store'
import { Slider } from './Slider'
import { exportPng, downloadBlob } from '../engine/export'
import { upscale, isUpscalerAvailable } from '../engine/upscale'
import type { UpscalerKind } from '../engine/models'

export function Controls() {
  const mode = useStore((s) => s.mode)
  if (mode === 'average') return <AveragePanel />
  if (mode === 'morph') return <MorphPanel />
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
    faces.map((f) => `${f.id}:${f.weight}:${f.enabled}:${!!f.landmarks}`),
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
        <div className="flex gap-2">
          {(['blur', 'studio', 'transparent'] as const).map((b) => (
            <button
              key={b}
              onClick={() => update({ background: b })}
              className={`btn text-xs flex-1 capitalize ${
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

function EnhancePanel() {
  const result = useStore((s) => s.result)
  const setResult = useStore((s) => s.setResult)
  const [kind, setKind] = useState<UpscalerKind>('photo')
  const [progress, setProgress] = useState<number | null>(null)
  const [msg, setMsg] = useState<string | null>(null)

  const run = async () => {
    if (!result) return
    setMsg(null)
    if (!(await isUpscalerAvailable(kind))) {
      setMsg('Could not reach the upscaler model — check your connection and retry.')
      return
    }
    setMsg('Loading model & upscaling… first run downloads the model.')
    setProgress(0)
    try {
      const up = await upscale(result, kind, (f) => setProgress(f))
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
        {progress !== null ? `Upscaling ${Math.round(progress * 100)}%` : 'Upscale 4×'}
      </button>
      {msg && <p className="text-xs text-muted">{msg}</p>}
      <p className="text-[11px] text-faint">
        Runs on-device via ONNX (WebGPU when available). Large images may take a while.
      </p>
    </div>
  )
}
