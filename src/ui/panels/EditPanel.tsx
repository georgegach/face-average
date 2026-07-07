import { useEffect, useState } from 'react'
import { useStore } from '../../state/store'
import { Slider } from '../Slider'
import { Icon } from '../Icon'
import { isAgingAvailable } from '../../engine/aging'
import { DEFAULT_EDIT_SETTINGS, type EditSettings } from '../../engine/types'
import { ResultActions, useAutoRerun } from './shared'

function Swatches({
  label,
  value,
  colors,
  onChange,
}: {
  label: string
  value: string | null
  colors: [string, string][] // [name, hex]
  onChange: (hex: string | null) => void
}) {
  return (
    <div>
      <div className="label mb-1">{label}</div>
      <div className="flex flex-wrap items-center gap-1.5">
        <button
          title="Off"
          aria-label={`${label} off`}
          onClick={() => onChange(null)}
          className={`w-6 h-6 rounded-full border text-[10px] leading-none grid place-items-center ${
            value === null
              ? 'border-accent text-accent-hi bg-accent/15'
              : 'border-edge text-muted hover:text-content'
          }`}
        >
          ×
        </button>
        {colors.map(([name, hex]) => (
          <button
            key={hex}
            title={name}
            aria-label={`${label} ${name}`}
            onClick={() => onChange(hex)}
            style={{ backgroundColor: hex }}
            className={`w-6 h-6 rounded-full border ${
              value === hex ? 'ring-2 ring-accent border-transparent' : 'border-edge'
            }`}
          />
        ))}
      </div>
    </div>
  )
}

const LIP_COLORS: [string, string][] = [
  ['Classic red', '#b3273b'],
  ['Coral', '#d96459'],
  ['Berry', '#8e3457'],
  ['Nude', '#b97a63'],
  ['Plum', '#6f2e4e'],
]

const EYE_COLORS: [string, string][] = [
  ['Blue', '#5b7fa6'],
  ['Green', '#5e7d54'],
  ['Hazel', '#8a6a42'],
  ['Amber', '#a97b32'],
  ['Grey', '#7d838c'],
]

const AGE_PRESETS: [string, number][] = [
  ['Child', 8],
  ['Young', 25],
  ['Old', 70],
]

// One-tap "Looks": each applies a bundle of edit settings over a clean baseline.
const LOOKS: [string, Partial<EditSettings>][] = [
  ['Natural', { skinSmooth: 0.35, teethWhiten: 0.2, blush: 0.15, browDefine: 0.2 }],
  [
    'Glam',
    {
      skinSmooth: 0.6,
      teethWhiten: 0.5,
      lipColor: '#b3273b',
      lipStrength: 0.7,
      blush: 0.35,
      browDefine: 0.4,
    },
  ],
  ['Studio', { skinSmooth: 0.4, background: 'studio', backgroundStrength: 0.7, vignette: 0.25 }],
  ['Bokeh', { skinSmooth: 0.3, background: 'bokeh', backgroundStrength: 0.75 }],
]

const HAIR_COLORS: [string, string][] = [
  ['Black', '#191a1e'],
  ['Dark brown', '#3b2a20'],
  ['Chestnut', '#6a4630'],
  ['Auburn', '#7c3b24'],
  ['Blonde', '#c9a35c'],
  ['Platinum', '#d9cdb8'],
  ['Silver', '#a7a9ad'],
  ['Pink', '#c96a94'],
  ['Blue', '#3d5a80'],
  ['Purple', '#5e3b76'],
]

export function EditPanel() {
  const faces = useStore((s) => s.faces)
  const editFaceId = useStore((s) => s.editFaceId)
  const setEditFace = useStore((s) => s.setEditFace)
  const setMode = useStore((s) => s.setMode)
  const es = useStore((s) => s.editSettings)
  const update = useStore((s) => s.updateEditSettings)
  const resetEditSettings = useStore((s) => s.resetEditSettings)
  const runEdit = useStore((s) => s.runEdit)
  const computing = useStore((s) => s.computing)
  const result = useStore((s) => s.result)
  const parseLoad = useStore((s) => s.parseLoad)
  const ageLoad = useStore((s) => s.ageLoad)
  const ageProgress = useStore((s) => s.ageProgress)
  const [ageAvailable, setAgeAvailable] = useState<boolean | null>(null)

  useEffect(() => {
    isAgingAvailable().then(setAgeAvailable)
  }, [])

  const usable = faces.filter((f) => f.landmarks && !f.failed)
  const face = usable.find((f) => f.id === editFaceId) ?? usable[0]

  // Re-render automatically when any tool changes, after the first result.
  const sig = JSON.stringify([es, face?.id])
  useAutoRerun(sig, !!result, runEdit)

  return (
    <div className="flex flex-col gap-4">
      <button className="btn-accent" disabled={!face || computing} onClick={runEdit}>
        {computing ? 'Applying…' : 'Apply edits'}
      </button>

      {parseLoad.loading && (
        <div className="card p-2">
          <div className="flex justify-between text-[10px] text-muted mb-1">
            <span>Downloading face parser…</span>
            <span>{Math.round(parseLoad.frac * 100)}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-surface3 overflow-hidden">
            <div
              className="h-full bg-accent transition-[width] duration-150"
              style={{ width: `${Math.round(parseLoad.frac * 100)}%` }}
            />
          </div>
        </div>
      )}

      <div>
        <div className="label mb-1">Face</div>
        <select
          value={face?.id ?? ''}
          onChange={(e) => setEditFace(e.target.value)}
          className="w-full px-3 py-2 text-sm"
        >
          <option value="" disabled>
            choose…
          </option>
          {usable.map((f) => (
            <option key={f.id} value={f.id}>
              {f.name}
            </option>
          ))}
        </select>
        {usable.length === 0 && (
          <p className="text-xs text-muted mt-2">Add a photo with a detected face first.</p>
        )}
      </div>

      <div>
        <div className="label mb-1">Looks</div>
        <div className="grid grid-cols-2 gap-2">
          {LOOKS.map(([name, patch]) => (
            <button
              key={name}
              className="btn btn-ghost text-xs"
              onClick={() => update({ ...DEFAULT_EDIT_SETTINGS, ...patch })}
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-3 card p-3">
        <div className="label">Retouch</div>
        <Slider
          label="Skin smooth"
          value={es.skinSmooth}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => update({ skinSmooth: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
        <Slider
          label="Teeth whiten"
          value={es.teethWhiten}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => update({ teethWhiten: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
      </div>

      <div className="flex flex-col gap-3 card p-3">
        <div className="label">Makeup</div>
        <Swatches
          label="Lips"
          value={es.lipColor}
          colors={LIP_COLORS}
          onChange={(hex) => update({ lipColor: hex })}
        />
        {es.lipColor && (
          <Slider
            label="Lip strength"
            value={es.lipStrength}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => update({ lipStrength: v })}
            format={(v) => `${Math.round(v * 100)}%`}
          />
        )}
        <Slider
          label="Blush"
          value={es.blush}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => update({ blush: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
        <Slider
          label="Brow define"
          value={es.browDefine}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => update({ browDefine: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
        <Swatches
          label="Eye colour"
          value={es.eyeColor}
          colors={EYE_COLORS}
          onChange={(hex) => update({ eyeColor: hex })}
        />
        {es.eyeColor && (
          <Slider
            label="Eye strength"
            value={es.eyeStrength}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => update({ eyeStrength: v })}
            format={(v) => `${Math.round(v * 100)}%`}
          />
        )}
      </div>

      <div className="flex flex-col gap-3 card p-3">
        <div className="label">Hair</div>
        <Swatches
          label="Hair colour"
          value={es.hairColor}
          colors={HAIR_COLORS}
          onChange={(hex) => update({ hairColor: hex })}
        />
        {es.hairColor && (
          <Slider
            label="Hair strength"
            value={es.hairStrength}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => update({ hairStrength: v })}
            format={(v) => `${Math.round(v * 100)}%`}
          />
        )}
      </div>

      <div className="flex flex-col gap-3 card p-3">
        <div className="label">Shape</div>
        <Slider
          label="Smile"
          value={es.smile}
          min={-1}
          max={1}
          step={0.05}
          onChange={(v) => update({ smile: v })}
          format={(v) => `${v > 0 ? '+' : ''}${Math.round(v * 100)}%`}
        />
        <Slider
          label="Eye size"
          value={es.eyeSize}
          min={-1}
          max={1}
          step={0.05}
          onChange={(v) => update({ eyeSize: v })}
          format={(v) => `${v > 0 ? '+' : ''}${Math.round(v * 100)}%`}
        />
        <Slider
          label="Nose slim"
          value={es.noseSlim}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => update({ noseSlim: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
        <Slider
          label="Face slim"
          value={es.faceSlim}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => update({ faceSlim: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
        <Slider
          label="Hair volume"
          value={es.hairVolume}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => update({ hairVolume: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
      </div>

      <div className="flex flex-col gap-3 card p-3">
        <div className="label">Age</div>
        {ageAvailable === false ? (
          <p className="text-xs text-muted">
            The re-aging model isn't published yet — run the convert-fran workflow once to
            enable this tool.
          </p>
        ) : (
          <>
            <label className="flex items-center justify-between text-sm">
              <span className="text-content">Re-age face</span>
              <input
                type="checkbox"
                checked={es.ageEnabled}
                onChange={(e) => update({ ageEnabled: e.target.checked })}
              />
            </label>
            <div className="flex gap-2">
              {AGE_PRESETS.map(([name, age]) => (
                <button
                  key={name}
                  onClick={() => update({ ageEnabled: true, targetAge: age })}
                  className={`btn text-xs flex-1 ${
                    es.ageEnabled && es.targetAge === age ? 'btn-accent' : 'btn-ghost'
                  }`}
                >
                  {name}
                </button>
              ))}
            </div>
            {es.ageEnabled && (
              <>
                <Slider
                  label="Current age"
                  value={es.sourceAge}
                  min={10}
                  max={90}
                  step={1}
                  onChange={(v) => update({ sourceAge: v })}
                  format={(v) => `${Math.round(v)}`}
                />
                <Slider
                  label="Target age"
                  value={es.targetAge}
                  min={5}
                  max={85}
                  step={1}
                  onChange={(v) => update({ targetAge: v })}
                  format={(v) => `${Math.round(v)}`}
                />
                <p className="text-[11px] text-faint">
                  One-time ~124 MB download. Fast with WebGPU; slower on CPU-only browsers.
                </p>
              </>
            )}
            {ageLoad.loading && (
              <div>
                <div className="flex justify-between text-[10px] text-muted mb-1">
                  <span>Downloading re-aging model…</span>
                  <span>{Math.round(ageLoad.frac * 100)}%</span>
                </div>
                <div className="h-1.5 rounded-full bg-surface3 overflow-hidden">
                  <div
                    className="h-full bg-accent transition-[width] duration-150"
                    style={{ width: `${Math.round(ageLoad.frac * 100)}%` }}
                  />
                </div>
              </div>
            )}
            {ageProgress !== null && (
              <div>
                <div className="flex justify-between text-[10px] text-muted mb-1">
                  <span>Re-aging…</span>
                  <span>{Math.round(ageProgress * 100)}%</span>
                </div>
                <div className="h-1.5 rounded-full bg-surface3 overflow-hidden">
                  <div
                    className="h-full bg-accent transition-[width] duration-150"
                    style={{ width: `${Math.round(ageProgress * 100)}%` }}
                  />
                </div>
              </div>
            )}
          </>
        )}
      </div>

      <div className="flex flex-col gap-3 card p-3">
        <div className="label">Scene</div>
        <div>
          <div className="label mb-1">Background</div>
          <div className="flex gap-2">
            {(['none', 'bokeh', 'studio'] as const).map((b) => (
              <button
                key={b}
                onClick={() => update({ background: b })}
                className={`btn text-xs flex-1 capitalize ${
                  es.background === b ? 'btn-accent' : 'btn-ghost'
                }`}
              >
                {b}
              </button>
            ))}
          </div>
        </div>
        {es.background !== 'none' && (
          <Slider
            label="Background strength"
            value={es.backgroundStrength}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => update({ backgroundStrength: v })}
            format={(v) => `${Math.round(v * 100)}%`}
          />
        )}
        <Slider
          label="Vignette"
          value={es.vignette}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => update({ vignette: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
      </div>

      <button className="text-xs text-muted hover:text-content text-left" onClick={resetEditSettings}>
        Reset all tools
      </button>

      {result && <ResultActions result={result} prefix="facestudio-edit" />}

      <button
        onClick={() => setMode('baby')}
        className="card p-3 flex items-center gap-2.5 text-left hover:border-accent/50 transition-colors"
      >
        <Icon name="heart" size={18} className="text-accent shrink-0" />
        <span className="flex-1 min-w-0">
          <span className="block text-xs font-semibold text-content">Try Future Baby</span>
          <span className="block text-[11px] text-muted">Blend two faces into a future child — just for fun.</span>
        </span>
        <span className="text-xs text-accent-hi shrink-0">→</span>
      </button>

      <p className="text-[11px] text-faint">
        All edits are mask-based pixel operations at full resolution — nothing is regenerated.
      </p>
    </div>
  )
}
