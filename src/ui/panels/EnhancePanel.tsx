import { useState } from 'react'
import { useStore } from '../../state/store'
import { upscale, type UpscaleStage } from '../../engine/upscale'
import type { UpscalerKind } from '../../engine/models'
import { tracked } from '../../lib/analytics'

export function EnhancePanel() {
  const result = useStore((s) => s.result)
  const setResult = useStore((s) => s.setResult)
  const [kind, setKind] = useState<UpscalerKind>('photo')
  const [progress, setProgress] = useState<number | null>(null)
  const [stage, setStage] = useState<UpscaleStage>('download')
  const [msg, setMsg] = useState<string | null>(null)

  const run = async () => {
    if (!result) return
    setMsg(null)
    setStage('download')
    setProgress(0)
    try {
      const up = await tracked('enhance', { model: kind }, () =>
        upscale(result, kind, (st, f) => {
          setStage(st)
          setProgress(f)
        }),
      )
      setResult(up)
      setMsg('Upscaled 4×.')
    } catch (e) {
      setMsg('Upscale failed: ' + (e as Error).message)
    } finally {
      setProgress(null)
    }
  }

  const kinds: [UpscalerKind, string][] = [
    ['photo', 'Photo'],
    ['anime', 'Anime / Art'],
    ['general', 'General / Nature'],
  ]

  return (
    <div className="flex flex-col gap-4">
      {!result && <p className="text-xs text-muted">Render an average or morph first.</p>}
      <div>
        <div className="label mb-1">Model</div>
        <div className="flex flex-col gap-2">
          {kinds.map(([k, name]) => (
            <button
              key={k}
              onClick={() => setKind(k)}
              className={`btn text-xs ${kind === k ? 'btn-accent' : 'btn-ghost'}`}
            >
              {name}
            </button>
          ))}
        </div>
      </div>
      <button className="btn-accent" disabled={!result || progress !== null} onClick={run}>
        {progress !== null
          ? `${stage === 'download' ? 'Downloading model' : 'Upscaling'} ${Math.round(progress * 100)}%`
          : 'Upscale 4×'}
      </button>
      {progress !== null && (
        <div className="h-1.5 rounded-full bg-surface3 overflow-hidden">
          <div
            className="h-full bg-accent transition-[width] duration-150"
            style={{ width: `${Math.round(progress * 100)}%` }}
          />
        </div>
      )}
      {msg && <p className="text-xs text-muted">{msg}</p>}
      <p className="text-[11px] text-faint">
        Runs on-device via ONNX (WebGPU when available). Large images may take a while.
      </p>
    </div>
  )
}
