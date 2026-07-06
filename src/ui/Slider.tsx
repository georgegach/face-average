import type { CSSProperties } from 'react'

interface Props {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  format?: (v: number) => string
}

export function Slider({ label, value, min, max, step = 1, onChange, format }: Props) {
  // Drives the iOS-style filled portion of the track (see index.css).
  const fill = `${((value - min) / (max - min)) * 100}%`
  return (
    <label className="block">
      <div className="flex items-center justify-between mb-1.5">
        <span className="label">{label}</span>
        <span className="text-xs text-content tabular-nums">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        className="w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ '--fill': fill } as CSSProperties}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  )
}
