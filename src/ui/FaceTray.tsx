import { useRef, useState, type CSSProperties } from 'react'
import { useStore } from '../state/store'
import { Icon } from './Icon'
import { PresetGallery } from './PresetGallery'

function MiniProgress({ label, frac, count }: { label: string; frac: number; count?: string }) {
  return (
    <div className="card p-2">
      <div className="flex justify-between text-[10px] text-muted mb-1">
        <span>{label}</span>
        {count ? <span>{count}</span> : frac >= 0 && <span>{Math.round(frac * 100)}%</span>}
      </div>
      <div className="h-1 rounded-full bg-surface3 overflow-hidden">
        {frac < 0 ? (
          <div className="h-full w-1/3 bg-accent animate-pulse" />
        ) : (
          <div
            className="h-full bg-accent transition-[width] duration-150 rounded-full"
            style={{ width: `${Math.round(frac * 100)}%` }}
          />
        )}
      </div>
    </div>
  )
}

export function FaceTray({
  onWebcam,
  onEdit,
}: {
  onWebcam: () => void
  onEdit: (id: string) => void
}) {
  const faces = useStore((s) => s.faces)
  const mode = useStore((s) => s.mode)
  const templateId = useStore((s) => s.settings.templateId)
  const addFiles = useStore((s) => s.addFiles)
  const removeFace = useStore((s) => s.removeFace)
  const setWeight = useStore((s) => s.setWeight)
  const toggleEnabled = useStore((s) => s.toggleEnabled)
  const setTemplate = useStore((s) => s.setTemplate)
  const clearFaces = useStore((s) => s.clearFaces)
  const modelLoad = useStore((s) => s.modelLoad)
  const detectQueue = useStore((s) => s.detectQueue)
  const inputRef = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)

  return (
    <div className="flex flex-col gap-3 h-full">
      <div className="flex items-center justify-between px-1">
        <h2 className="text-sm font-semibold text-content flex items-center gap-2">
          Faces
          {faces.length > 0 && (
            <span className="pill !px-2 !py-0.5 text-muted tabular-nums">{faces.length}</span>
          )}
        </h2>
        {faces.length > 0 && (
          <button className="text-xs text-muted hover:text-content" onClick={clearFaces}>
            Clear all
          </button>
        )}
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDrag(true)
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDrag(false)
          addFiles(e.dataTransfer.files)
        }}
        onClick={() => inputRef.current?.click()}
        className={`dropzone text-center py-5 px-3 ${drag ? 'border-accent bg-accent/5' : ''}`}
      >
        <Icon name="upload" size={20} className="mx-auto text-accent mb-1.5" />
        <p className="text-sm font-medium text-content">Add faces</p>
        <p className="text-xs text-muted mt-0.5">Drop images or click · JPG/PNG · batch OK</p>
        <button
          className="btn-ghost mt-3 text-xs py-1.5"
          onClick={(e) => {
            e.stopPropagation()
            onWebcam()
          }}
        >
          <Icon name="camera" size={14} /> Use webcam
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => e.target.files && addFiles(e.target.files)}
        />
      </div>

      {modelLoad.loading && (
        <MiniProgress
          label={modelLoad.frac < 0 ? 'Preparing face engine…' : 'Downloading face model…'}
          frac={modelLoad.frac}
        />
      )}

      {detectQueue && detectQueue.total > 1 && (
        <MiniProgress
          label="Detecting faces…"
          frac={detectQueue.done / detectQueue.total}
          count={`${detectQueue.done}/${detectQueue.total}`}
        />
      )}

      <div className="md:flex-1 overflow-y-auto flex flex-col gap-2 pr-1 max-h-[42vh] md:max-h-none">
        {faces.map((f) => (
          <div key={f.id} className="card p-2 flex gap-2.5 items-start">
            <img
              src={f.thumb}
              alt={f.name}
              className={`w-14 h-14 rounded-xl object-cover transition-opacity ${
                f.enabled ? '' : 'opacity-30'
              }`}
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-medium text-content truncate flex-1">{f.name}</span>
                {!f.detecting && !f.failed && f.landmarks && (
                  <button
                    title="Edit landmarks"
                    aria-label="Edit landmarks"
                    className={`hover:text-content ${f.editRev > 0 ? 'text-accent-hi' : 'text-muted'}`}
                    onClick={() => onEdit(f.id)}
                  >
                    <Icon name="edit" size={13} />
                  </button>
                )}
                <button
                  title="Remove"
                  aria-label="Remove face"
                  className="text-muted hover:text-red-400"
                  onClick={() => removeFace(f.id)}
                >
                  <Icon name="close" size={13} />
                </button>
              </div>
              <div className="text-[10px] mt-0.5">
                {f.detecting ? (
                  <span className="text-accent-hi">detecting…</span>
                ) : f.failed ? (
                  <span className="text-red-400">no face found</span>
                ) : (
                  <span className="text-emerald-500 inline-flex items-center gap-1">
                    <Icon name="check" size={11} /> 478 pts
                  </span>
                )}
              </div>
              {mode === 'average' && !f.failed && (
                <div className="flex items-center gap-2 mt-1.5">
                  <input
                    type="range"
                    min={0}
                    max={3}
                    step={0.1}
                    value={f.weight}
                    onChange={(e) => setWeight(f.id, parseFloat(e.target.value))}
                    className="flex-1"
                    style={{ '--fill': `${(f.weight / 3) * 100}%` } as CSSProperties}
                    disabled={!f.enabled}
                    aria-label={`Weight of ${f.name}`}
                  />
                  <button
                    title="Use as template shape"
                    aria-label="Use as template shape"
                    onClick={() => setTemplate(templateId === f.id ? null : f.id)}
                    className={templateId === f.id ? 'text-accent-hi' : 'text-muted hover:text-content'}
                  >
                    <Icon name="star" size={14} filled={templateId === f.id} />
                  </button>
                  <button
                    onClick={() => toggleEnabled(f.id)}
                    className={f.enabled ? 'text-emerald-500' : 'text-faint'}
                    title="Enable/disable"
                    aria-label="Enable or disable face"
                  >
                    <Icon name="power" size={14} />
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
        {faces.length > 0 && (
          <div className="mt-1">
            <PresetGallery compact />
          </div>
        )}
      </div>
    </div>
  )
}
