import { useEffect } from 'react'
import { useStore } from '../../state/store'
import { Icon } from '../Icon'
import { exportPng, downloadBlob, shareImage } from '../../engine/export'
import { capture } from '../../lib/analytics'

/** Shared result actions: download the PNG (1× / 2×) and share the rendered image. */
export function ResultActions({ result, prefix }: { result: ImageData; prefix: string }) {
  const doExport = async (scale: number) => {
    capture('export', { format: 'png', scale, tool: prefix })
    downloadBlob(await exportPng(result, scale), `${prefix}${scale > 1 ? '@2x' : ''}.png`)
  }
  const doShare = async () => {
    capture('share', { tool: prefix })
    shareImage(await exportPng(result, 1), `${prefix}.png`, 'Made with FaceStudio — 100% on-device')
  }
  const canShare = typeof navigator !== 'undefined' && 'share' in navigator
  return (
    <div className="flex gap-2 border-t border-edge/70 pt-3">
      <button className="btn-ghost flex-1 text-xs" onClick={() => doExport(1)}>
        PNG
      </button>
      <button className="btn-ghost flex-1 text-xs" onClick={() => doExport(2)}>
        PNG @2×
      </button>
      {canShare && (
        <button className="btn-ghost flex-1 text-xs" onClick={doShare}>
          <Icon name="share" size={13} /> Share
        </button>
      )}
    </div>
  )
}

export function OutputSize() {
  const settings = useStore((s) => s.settings)
  const update = useStore((s) => s.updateSettings)
  const sizes: [string, number, number][] = [
    ['Portrait', 768, 1024],
    ['Square', 1024, 1024],
    ['Large', 1024, 1365],
  ]
  return (
    <div>
      <div className="label mb-1">Output size</div>
      <div className="flex gap-2">
        {sizes.map(([name, w, h]) => (
          <button
            key={name}
            onClick={() => update({ outWidth: w, outHeight: h })}
            className={`btn text-xs flex-1 ${
              settings.outWidth === w && settings.outHeight === h ? 'btn-accent' : 'btn-ghost'
            }`}
          >
            {name}
          </button>
        ))}
      </div>
    </div>
  )
}

/** Debounced auto re-run after the first result, keyed on a settings signature. */
export function useAutoRerun(sig: string, enabled: boolean, run: () => void, delay = 140) {
  useEffect(() => {
    if (!enabled) return
    const t = setTimeout(run, delay)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig])
}
