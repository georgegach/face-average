import { useState } from 'react'
import { useStore, type Mode } from '../state/store'
import { FaceTray } from '../ui/FaceTray'
import { Stage } from '../ui/Stage'
import { Controls } from '../ui/Controls'
import { WebcamModal } from '../ui/Webcam'
import { LandmarkEditor } from '../ui/LandmarkEditor'
import { Icon, type IconName } from '../ui/Icon'
import { getTheme, setTheme, type Theme } from '../ui/theme'

const MODE_META: Record<Mode, { label: string; icon: IconName; blurb: string }> = {
  edit: {
    label: 'Edit',
    icon: 'wand',
    blurb: 'Retouch, makeup, reshape, restyle hair and re-age — pro face editing, 100% on your device.',
  },
  baby: {
    label: 'Future Baby',
    icon: 'heart',
    blurb: 'Blend two parents and de-age the result to glimpse your future child — just for fun.',
  },
  average: {
    label: 'Average',
    icon: 'faces',
    blurb: 'Blend any number of faces into one composite, with per-face weights.',
  },
  morph: {
    label: 'Morph',
    icon: 'blend',
    blurb: 'Transition between two faces and export the animation as video or GIF.',
  },
  replace: {
    label: 'Replace',
    icon: 'swap',
    blurb: 'Swap the face in a photo using the closest-pose source — no AI generation.',
  },
  enhance: {
    label: 'Enhance',
    icon: 'sparkles',
    blurb: 'Upscale the current result 4× with on-device Real-ESRGAN.',
  },
}

// The editor is the flagship; everything else is a secondary tool behind "More tools".
const SECONDARY_MODES: Mode[] = ['baby', 'average', 'morph', 'replace', 'enhance']

export default function App() {
  const mode = useStore((s) => s.mode)
  const setMode = useStore((s) => s.setMode)
  const [webcam, setWebcam] = useState(false)
  const [editId, setEditId] = useState<string | null>(null)
  const [toolsOpen, setToolsOpen] = useState(false)
  const [theme, setThemeState] = useState<Theme>(getTheme())
  const secondaryActive = mode !== 'edit'

  const toggleTheme = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    setThemeState(next)
  }

  return (
    <div className="flex flex-col min-h-full md:h-full">
      <header className="sticky top-0 z-20 px-3 sm:px-4 pt-3 flex flex-col gap-2">
        <div className="panel flex items-center gap-3 px-4 py-2">
          <div className="flex items-center gap-2 shrink-0">
            <Icon name="logo" size={22} className="text-accent" />
            <span className="font-semibold tracking-tight text-[15px]">FaceStudio</span>
          </div>
          <span className="pill text-muted hidden sm:inline-flex" title="Nothing is uploaded — all processing happens in your browser">
            <Icon name="shield" size={12} className="text-emerald-500" />
            100% on-device
          </span>
          <div className="flex-1" />
          <button
            onClick={toggleTheme}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-label="Toggle theme"
            className="btn-ghost !p-2 text-muted hover:text-content"
          >
            <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={16} />
          </button>
          <a
            href="https://github.com/georgegach/face-average"
            target="_blank"
            rel="noreferrer"
            title="View source on GitHub"
            aria-label="View source on GitHub"
            className="btn-ghost !p-2 text-muted hover:text-content"
          >
            <Icon name="github" size={16} />
          </a>
        </div>
        {/* Mode switcher: the flagship Edit segment + a "More tools" popover for the rest. */}
        <nav className="relative flex justify-center pb-1">
          <div className="seg panel !rounded-full !p-1">
            <button
              onClick={() => {
                setMode('edit')
                setToolsOpen(false)
              }}
              aria-current={mode === 'edit' ? 'page' : undefined}
              className={`seg-item ${mode === 'edit' ? 'seg-item-active' : ''}`}
            >
              <Icon name="wand" size={15} className={mode === 'edit' ? 'text-accent' : ''} />
              Edit
            </button>
            <button
              onClick={() => setToolsOpen((o) => !o)}
              aria-expanded={toolsOpen}
              aria-current={secondaryActive ? 'page' : undefined}
              className={`seg-item ${secondaryActive ? 'seg-item-active' : ''}`}
            >
              <Icon
                name={secondaryActive ? MODE_META[mode].icon : 'sparkles'}
                size={15}
                className={secondaryActive ? 'text-accent' : ''}
              />
              {secondaryActive ? MODE_META[mode].label : 'More tools'}
              <span className="text-[10px] opacity-60 ml-0.5">▾</span>
            </button>
          </div>
          {toolsOpen && (
            <>
              {/* click-away backdrop */}
              <div className="fixed inset-0 z-30" onClick={() => setToolsOpen(false)} />
              <div className="absolute top-full mt-2 z-40 panel !rounded-2xl p-1 flex flex-col gap-0.5 min-w-[210px]">
                {SECONDARY_MODES.map((m) => (
                  <button
                    key={m}
                    onClick={() => {
                      setMode(m)
                      setToolsOpen(false)
                    }}
                    aria-current={mode === m ? 'page' : undefined}
                    className={`seg-item !justify-start !rounded-xl w-full ${
                      mode === m ? 'seg-item-active' : ''
                    }`}
                  >
                    <Icon
                      name={MODE_META[m].icon}
                      size={15}
                      className={mode === m ? 'text-accent' : ''}
                    />
                    {MODE_META[m].label}
                    {m === 'baby' && (
                      <span className="pill !px-2 !py-0.5 ml-auto text-[10px] text-accent-hi">fun</span>
                    )}
                  </button>
                ))}
              </div>
            </>
          )}
        </nav>
      </header>

      <main className="flex-1 flex flex-col gap-3 p-3 sm:px-4 md:grid md:grid-cols-[264px_1fr_292px] md:min-h-0 md:overflow-hidden">
        <aside className="panel p-3 order-2 md:order-1 md:min-h-0 md:overflow-hidden">
          <FaceTray onWebcam={() => setWebcam(true)} onEdit={setEditId} />
        </aside>
        <section className="panel flex flex-col order-1 md:order-2 min-h-[55vh] md:min-h-0 overflow-hidden">
          <Stage />
        </section>
        <aside className="panel p-4 order-3 md:min-h-0 md:overflow-y-auto">
          <div className="mb-4">
            <h2 className="text-sm font-semibold text-content flex items-center gap-1.5">
              <Icon name={MODE_META[mode].icon} size={15} className="text-accent" />
              {MODE_META[mode].label}
            </h2>
            <p className="text-xs text-muted mt-1">{MODE_META[mode].blurb}</p>
          </div>
          <Controls />
        </aside>
      </main>

      <footer className="px-4 py-3 text-center text-[11px] text-faint">
        100% local · your photos never leave your device · anonymous usage analytics only
      </footer>

      {webcam && <WebcamModal onClose={() => setWebcam(false)} />}
      {editId && <LandmarkEditor faceId={editId} onClose={() => setEditId(null)} />}
    </div>
  )
}
