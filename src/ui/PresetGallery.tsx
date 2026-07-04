import { useEffect, useState } from 'react'
import { useStore } from '../state/store'
import { Icon } from './Icon'

interface Gallery {
  id: string
  title: string
  subtitle: string
  images: string[]
}

export function PresetGallery({ compact = false }: { compact?: boolean }) {
  const [galleries, setGalleries] = useState<Gallery[]>([])
  const addFromUrls = useStore((s) => s.addFromUrls)
  const base = import.meta.env.BASE_URL

  useEffect(() => {
    fetch(`${base}presets/manifest.json`)
      .then((r) => r.json())
      .then((d) => setGalleries(d.galleries))
      .catch(() => setGalleries([]))
  }, [base])

  const load = (g: Gallery) => addFromUrls(g.images.map((p) => base + p))

  if (compact) {
    return (
      <div className="flex flex-wrap gap-2">
        {galleries.map((g) => (
          <button key={g.id} className="btn-ghost text-xs py-1" onClick={() => load(g)}>
            Try: {g.title}
          </button>
        ))}
      </div>
    )
  }

  return (
    <div className="max-w-md text-center">
      <div className="flex justify-center mb-3 text-accent-hi">
        <Icon name="faces" size={56} />
      </div>
      <h2 className="text-xl font-semibold text-content">Blend faces into one</h2>
      <p className="text-sm text-muted mt-2 mb-5">
        Drop a few portraits on the left to average them, or start with a preset. Everything runs in
        your browser — your photos never leave your device.
      </p>
      <div className="flex flex-col gap-2">
        {galleries.map((g) => (
          <button key={g.id} className="btn-accent" onClick={() => load(g)}>
            {g.title} — {g.subtitle}
          </button>
        ))}
      </div>
    </div>
  )
}
