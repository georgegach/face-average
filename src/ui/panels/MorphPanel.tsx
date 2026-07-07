import { useStore } from '../../state/store'
import { OutputSize } from './shared'

export function MorphPanel() {
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
