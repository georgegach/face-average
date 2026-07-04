import { useEffect, useRef, useState } from 'react'
import { useStore } from '../state/store'

export function WebcamModal({ onClose }: { onClose: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const addBitmap = useStore((s) => s.addBitmap)
  const [error, setError] = useState<string | null>(null)
  const [n, setN] = useState(0)

  useEffect(() => {
    let cancelled = false
    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: 'user', width: 1280, height: 720 } })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) videoRef.current.srcObject = stream
      })
      .catch(() => setError('Camera unavailable or permission denied.'))
    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const capture = async () => {
    const v = videoRef.current
    if (!v || !v.videoWidth) return
    const c = document.createElement('canvas')
    c.width = v.videoWidth
    c.height = v.videoHeight
    c.getContext('2d')!.drawImage(v, 0, 0)
    const bmp = await createImageBitmap(c)
    await addBitmap(bmp, `webcam-${n + 1}`)
    setN((x) => x + 1)
  }

  return (
    <div className="fixed inset-0 bg-black/70 grid place-items-center z-50 p-4" onClick={onClose}>
      <div className="panel p-4 max-w-lg w-full" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold">Webcam capture</h3>
          <button className="text-slate-400 hover:text-slate-200" onClick={onClose}>
            ✕
          </button>
        </div>
        {error ? (
          <p className="text-sm text-red-400 py-8 text-center">{error}</p>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full rounded-xl bg-ink-900 aspect-video object-cover"
            />
            <div className="flex gap-2 mt-3">
              <button className="btn-accent flex-1" onClick={capture}>
                Capture{n > 0 ? ` (${n})` : ''}
              </button>
              <button className="btn-ghost" onClick={onClose}>
                Done
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
