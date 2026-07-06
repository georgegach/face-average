import { useEffect, useState } from 'react'
import { useStore } from '../state/store'
import { Icon, type IconName } from './Icon'

interface Gallery {
  id: string
  title: string
  subtitle: string
  images: string[]
}

function useGalleries(): Gallery[] {
  const [galleries, setGalleries] = useState<Gallery[]>([])
  const base = import.meta.env.BASE_URL
  useEffect(() => {
    fetch(`${base}presets/manifest.json`)
      .then((r) => r.json())
      .then((d) => setGalleries(d.galleries))
      .catch(() => setGalleries([]))
  }, [base])
  return galleries
}

/** Preset card: overlapping face thumbnails + title, loads the whole set. */
function PresetCard({ gallery, onLoad }: { gallery: Gallery; onLoad: () => void }) {
  const base = import.meta.env.BASE_URL
  return (
    <button
      onClick={onLoad}
      className="card w-full flex items-center gap-3 p-3 text-left transition-all duration-150
        hover:border-accent/50 hover:-translate-y-0.5 active:translate-y-0"
    >
      <div className="flex shrink-0 -space-x-3">
        {gallery.images.slice(0, 4).map((img) => (
          <img
            key={img}
            src={base + img}
            alt=""
            loading="lazy"
            className="w-10 h-10 rounded-full object-cover border-2 border-surface shadow-sm"
          />
        ))}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-content">{gallery.title}</div>
        <div className="text-xs text-muted truncate">{gallery.subtitle}</div>
      </div>
      <span className="text-xs text-accent-hi font-medium shrink-0">
        Try it · {gallery.images.length} faces
      </span>
    </button>
  )
}

const STEPS: { icon: IconName; title: string; text: string }[] = [
  { icon: 'upload', title: 'Add faces', text: 'Drop photos, use the webcam, or load a preset.' },
  { icon: 'wand', title: 'Pick a tool', text: 'Average, morph, replace, edit or enhance.' },
  { icon: 'download', title: 'Export', text: 'Save as PNG, GIF or video — all offline.' },
]

export function PresetGallery({ compact = false }: { compact?: boolean }) {
  const galleries = useGalleries()
  const addFromUrls = useStore((s) => s.addFromUrls)
  const base = import.meta.env.BASE_URL

  const load = (g: Gallery) => addFromUrls(g.images.map((p) => base + p))

  if (compact) {
    return (
      <div className="flex flex-col gap-2">
        {galleries.map((g) => (
          <button key={g.id} className="btn-ghost text-xs py-1.5 justify-start" onClick={() => load(g)}>
            <Icon name="photo" size={13} />
            Try preset: {g.title}
          </button>
        ))}
      </div>
    )
  }

  // Full onboarding empty state.
  return (
    <div className="max-w-lg w-full text-center">
      <div className="flex justify-center mb-4">
        <div
          className="w-16 h-16 rounded-[22px] grid place-items-center text-white shadow-glass"
          style={{ background: 'linear-gradient(135deg, #0a84ff, #bf5af2)' }}
        >
          <Icon name="faces" size={34} />
        </div>
      </div>
      <h2 className="text-2xl font-semibold tracking-tight text-content">Blend faces into one</h2>
      <p className="text-sm text-muted mt-2">
        Average, morph, replace, retouch and upscale faces — entirely in your browser.
      </p>
      <p className="pill text-muted mt-3">
        <Icon name="shield" size={12} className="text-emerald-500" />
        Private by design — photos never leave your device
      </p>

      <div className="grid grid-cols-3 gap-2 mt-6 text-left">
        {STEPS.map((s, i) => (
          <div key={s.title} className="card p-3">
            <div className="flex items-center gap-1.5 text-accent-hi">
              <span className="text-[11px] font-semibold tabular-nums">{i + 1}</span>
              <Icon name={s.icon} size={14} />
            </div>
            <div className="text-xs font-semibold text-content mt-1.5">{s.title}</div>
            <div className="text-[11px] text-muted mt-0.5 leading-snug">{s.text}</div>
          </div>
        ))}
      </div>

      {galleries.length > 0 && (
        <div className="mt-6 flex flex-col gap-2">
          <div className="label text-left">Start with a preset</div>
          {galleries.map((g) => (
            <PresetCard key={g.id} gallery={g} onLoad={() => load(g)} />
          ))}
        </div>
      )}
    </div>
  )
}
