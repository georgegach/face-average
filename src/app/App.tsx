import { useState } from 'react'
import { useStore, type Mode } from '../state/store'
import { FaceTray } from '../ui/FaceTray'
import { Stage } from '../ui/Stage'
import { Controls } from '../ui/Controls'
import { WebcamModal } from '../ui/Webcam'
import { PresetGallery } from '../ui/PresetGallery'

const MODES: [Mode, string][] = [
  ['average', 'Average'],
  ['morph', 'Morph'],
  ['enhance', 'Enhance'],
]

export default function App() {
  const mode = useStore((s) => s.mode)
  const setMode = useStore((s) => s.setMode)
  const faceCount = useStore((s) => s.faces.length)
  const [webcam, setWebcam] = useState(false)

  return (
    <div className="h-full flex flex-col">
      <header className="flex items-center gap-4 px-4 py-3 border-b border-ink-700/60">
        <div className="flex items-center gap-2">
          <span className="text-accent text-xl">◉</span>
          <span className="font-semibold tracking-tight">FaceStudio</span>
        </div>
        <nav className="flex gap-1 ml-2">
          {MODES.map(([m, label]) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`tab ${mode === m ? 'tab-active' : ''}`}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="flex-1" />
        {faceCount > 0 && <PresetGallery compact />}
        <a
          href="https://github.com/georgegach/face-average"
          target="_blank"
          rel="noreferrer"
          className="text-slate-500 hover:text-slate-200 text-sm"
        >
          GitHub
        </a>
      </header>

      <main className="flex-1 grid grid-cols-1 md:grid-cols-[260px_1fr_280px] gap-3 p-3 min-h-0">
        <aside className="panel p-3 min-h-0 order-2 md:order-1">
          <FaceTray onWebcam={() => setWebcam(true)} />
        </aside>
        <section className="panel min-h-0 flex flex-col order-1 md:order-2 overflow-hidden">
          <Stage />
        </section>
        <aside className="panel p-4 min-h-0 order-3 overflow-y-auto">
          <h2 className="text-sm font-semibold text-slate-300 mb-4 capitalize">{mode}</h2>
          <Controls />
        </aside>
      </main>

      <footer className="px-4 py-2 text-center text-[11px] text-slate-600 border-t border-ink-700/60">
        100% local · your photos never leave your device · WASM + WebGL in your browser
      </footer>

      {webcam && <WebcamModal onClose={() => setWebcam(false)} />}
    </div>
  )
}
