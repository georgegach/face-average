import { useState } from 'react'
import { useStore, type Mode } from '../state/store'
import { FaceTray } from '../ui/FaceTray'
import { Stage } from '../ui/Stage'
import { Controls } from '../ui/Controls'
import { WebcamModal } from '../ui/Webcam'
import { PresetGallery } from '../ui/PresetGallery'
import { Icon } from '../ui/Icon'
import { getTheme, setTheme, type Theme } from '../ui/theme'

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
  const [theme, setThemeState] = useState<Theme>(getTheme())

  const toggleTheme = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    setThemeState(next)
  }

  return (
    <div className="flex flex-col min-h-full md:h-full">
      <header className="sticky top-0 z-20 bg-bg/90 backdrop-blur border-b border-edge/70">
        <div className="flex items-center gap-3 px-3 sm:px-4 py-2.5">
          <div className="flex items-center gap-2 shrink-0">
            <Icon name="logo" size={22} className="text-accent-hi" />
            <span className="font-semibold tracking-tight">FaceStudio</span>
          </div>
          <div className="flex-1" />
          {faceCount > 0 && (
            <div className="hidden sm:block">
              <PresetGallery compact />
            </div>
          )}
          <button
            onClick={toggleTheme}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-label="Toggle theme"
            className="text-muted hover:text-content shrink-0 leading-none"
          >
            <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={18} />
          </button>
          <a
            href="https://github.com/georgegach/face-average"
            target="_blank"
            rel="noreferrer"
            className="text-muted hover:text-content text-sm shrink-0"
          >
            GitHub
          </a>
        </div>
        {/* Mode tabs: full-width segmented control, scrollable if cramped. */}
        <nav className="flex gap-1 px-2 pb-2 overflow-x-auto">
          {MODES.map(([m, label]) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`tab flex-1 min-w-[84px] ${mode === m ? 'tab-active' : ''}`}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1 flex flex-col gap-3 p-3 md:grid md:grid-cols-[260px_1fr_280px] md:min-h-0 md:overflow-hidden">
        <aside className="panel p-3 order-2 md:order-1 md:min-h-0 md:overflow-hidden">
          <FaceTray onWebcam={() => setWebcam(true)} />
        </aside>
        <section className="panel flex flex-col order-1 md:order-2 min-h-[55vh] md:min-h-0 overflow-hidden">
          <Stage />
        </section>
        <aside className="panel p-4 order-3 md:min-h-0 md:overflow-y-auto">
          <h2 className="text-sm font-semibold text-content mb-4 capitalize">{mode}</h2>
          <Controls />
        </aside>
      </main>

      <footer className="px-4 py-3 text-center text-[11px] text-faint border-t border-edge/70">
        100% local · your photos never leave your device · WASM + WebGL in your browser
      </footer>

      {webcam && <WebcamModal onClose={() => setWebcam(false)} />}
    </div>
  )
}
