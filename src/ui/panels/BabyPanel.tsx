import { useEffect, useState } from 'react'
import { useStore } from '../../state/store'
import { Slider } from '../Slider'
import { isAgingAvailable } from '../../engine/aging'
import { ResultActions, useAutoRerun } from './shared'

export function BabyPanel() {
  const faces = useStore((s) => s.faces).filter((f) => f.landmarks && !f.failed)
  const babyA = useStore((s) => s.babyA)
  const babyB = useStore((s) => s.babyB)
  const setBabyParents = useStore((s) => s.setBabyParents)
  const bs = useStore((s) => s.babySettings)
  const update = useStore((s) => s.updateBabySettings)
  const runBaby = useStore((s) => s.runBaby)
  const computing = useStore((s) => s.computing)
  const result = useStore((s) => s.result)
  const ageLoad = useStore((s) => s.ageLoad)
  const [ageAvailable, setAgeAvailable] = useState<boolean | null>(null)

  useEffect(() => {
    isAgingAvailable().then(setAgeAvailable)
  }, [])

  const A = faces.find((f) => f.id === babyA) ?? faces[0]
  const B = faces.find((f) => f.id === babyB && f.id !== A?.id) ?? faces.find((f) => f.id !== A?.id)
  const ready = !!A && !!B

  // Re-render automatically when parents or settings change, after the first result.
  const sig = JSON.stringify([bs, A?.id, B?.id])
  useAutoRerun(sig, !!result, runBaby, 160)

  const picker = (label: string, value: string | null, set: (id: string) => void) => (
    <div>
      <div className="label mb-1">{label}</div>
      <select
        value={value ?? ''}
        onChange={(e) => set(e.target.value)}
        className="w-full px-3 py-2 text-sm"
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
      <button className="btn-accent" disabled={!ready || computing} onClick={runBaby}>
        {computing ? 'Generating…' : 'Generate baby'}
      </button>

      {faces.length < 2 && (
        <p className="text-xs text-muted">Add two parent photos with detected faces.</p>
      )}

      {picker('Parent A', A?.id ?? null, (id) => setBabyParents(id, B?.id ?? null))}
      {picker('Parent B', B?.id ?? null, (id) => setBabyParents(A?.id ?? null, id))}

      <div className="flex flex-col gap-3 card p-3">
        <Slider
          label="Resemblance"
          value={bs.parentLean}
          min={-1}
          max={1}
          step={0.05}
          onChange={(v) => update({ parentLean: v })}
          format={(v) =>
            v === 0
              ? '50 / 50'
              : v < 0
                ? `${Math.round(-v * 100)}% Parent A`
                : `${Math.round(v * 100)}% Parent B`
          }
        />
        <label className="flex items-center justify-between text-sm">
          <span className="text-content">De-age to child</span>
          <input
            type="checkbox"
            checked={bs.deAge}
            onChange={(e) => update({ deAge: e.target.checked })}
          />
        </label>
        {bs.deAge && (
          <Slider
            label="Child age"
            value={bs.childAge}
            min={5}
            max={12}
            step={1}
            onChange={(v) => update({ childAge: v })}
            format={(v) => `${Math.round(v)} yrs`}
          />
        )}
      </div>

      {ageAvailable === false && (
        <p className="text-xs text-muted">
          The de-aging model isn't published yet — showing a blended preview of both parents. Run
          the convert-fran workflow once to enable the child version.
        </p>
      )}

      {ageLoad.loading && (
        <div className="card p-2">
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

      {result && <ResultActions result={result} prefix="facestudio-baby" />}

      <p className="text-[11px] text-faint">
        Just for fun — blends both parents and de-ages the result on your device. Not a genetic
        prediction, and nothing is uploaded.
      </p>
    </div>
  )
}
