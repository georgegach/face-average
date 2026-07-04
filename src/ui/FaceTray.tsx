import { useRef, useState } from 'react'
import { useStore } from '../state/store'

export function FaceTray({ onWebcam }: { onWebcam: () => void }) {
  const faces = useStore((s) => s.faces)
  const mode = useStore((s) => s.mode)
  const templateId = useStore((s) => s.settings.templateId)
  const addFiles = useStore((s) => s.addFiles)
  const removeFace = useStore((s) => s.removeFace)
  const setWeight = useStore((s) => s.setWeight)
  const toggleEnabled = useStore((s) => s.toggleEnabled)
  const setTemplate = useStore((s) => s.setTemplate)
  const clearFaces = useStore((s) => s.clearFaces)
  const inputRef = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)

  return (
    <div className="flex flex-col gap-3 h-full">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-300">Faces ({faces.length})</h2>
        {faces.length > 0 && (
          <button className="text-xs text-slate-500 hover:text-slate-300" onClick={clearFaces}>
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
        className={`panel border-dashed cursor-pointer text-center py-6 px-3 transition-colors ${
          drag ? 'border-accent bg-accent/5' : ''
        }`}
      >
        <p className="text-sm text-slate-300">Drop faces or click</p>
        <p className="text-xs text-slate-500 mt-1">JP/PNG · batch ok</p>
        <button
          className="btn-ghost mt-3 text-xs py-1"
          onClick={(e) => {
            e.stopPropagation()
            onWebcam()
          }}
        >
          📷 Webcam
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

      <div className="flex-1 overflow-y-auto flex flex-col gap-2 pr-1">
        {faces.map((f) => (
          <div key={f.id} className="panel p-2 flex gap-2 items-start">
            <img
              src={f.thumb}
              alt={f.name}
              className={`w-14 h-14 rounded-lg object-cover ${f.enabled ? '' : 'opacity-30'}`}
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1">
                <span className="text-xs text-slate-200 truncate flex-1">{f.name}</span>
                <button
                  title="Remove"
                  className="text-slate-500 hover:text-red-400 text-xs"
                  onClick={() => removeFace(f.id)}
                >
                  ✕
                </button>
              </div>
              <div className="text-[10px] mt-0.5">
                {f.detecting ? (
                  <span className="text-accent">detecting…</span>
                ) : f.failed ? (
                  <span className="text-red-400">no face found</span>
                ) : (
                  <span className="text-emerald-400">478 pts ✓</span>
                )}
              </div>
              {mode === 'average' && !f.failed && (
                <div className="flex items-center gap-2 mt-1">
                  <input
                    type="range"
                    min={0}
                    max={3}
                    step={0.1}
                    value={f.weight}
                    onChange={(e) => setWeight(f.id, parseFloat(e.target.value))}
                    className="flex-1"
                    disabled={!f.enabled}
                  />
                  <button
                    title="Use as template shape"
                    onClick={() => setTemplate(templateId === f.id ? null : f.id)}
                    className={`text-xs ${templateId === f.id ? 'text-accent' : 'text-slate-500 hover:text-slate-300'}`}
                  >
                    ★
                  </button>
                  <button
                    onClick={() => toggleEnabled(f.id)}
                    className={`text-xs ${f.enabled ? 'text-emerald-400' : 'text-slate-600'}`}
                    title="Enable/disable"
                  >
                    ⏻
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
