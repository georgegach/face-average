import { useState } from 'react'
import { useStore } from '../../state/store'
import { Slider } from '../Slider'
import { OutputSize, ResultActions, useAutoRerun } from './shared'

export function AveragePanel() {
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
  useAutoRerun(sig, hasResult, runAverage)

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
        <div className="flex flex-col gap-3 card p-3">
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

      {result && <ResultActions result={result} prefix="facestudio-average" />}
    </div>
  )
}
