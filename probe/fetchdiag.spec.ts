import { test } from '@playwright/test'

const URL_MODEL =
  'https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x-ClearRealityV1.onnx'

test('model fetch + Cache API round-trip', async ({ page }) => {
  await page.goto('/')
  // Ensure the service worker is controlling the page (matches real usage).
  await page.reload()
  await page.waitForTimeout(1500)

  const r = await page.evaluate(async (u) => {
    const out: Record<string, unknown> = {}
    out.swController = !!navigator.serviceWorker?.controller
    // Mirror engine/download.ts:fetchWithProgressCached
    const cache = await caches.open('facestudio-model-cache-v1')
    const before = await cache.match(u)
    out.preCached = !!before
    const res = await fetch(u)
    out.fetchStatus = res.status
    const buf = await res.arrayBuffer()
    out.bytes = buf.byteLength
    await cache.put(
      u,
      new Response(buf, { headers: { 'Content-Length': String(buf.byteLength) } }),
    )
    const hit = await cache.match(u)
    out.cacheHit = !!hit
    out.cacheBytes = hit ? (await hit.arrayBuffer()).byteLength : 0
    return out
  }, URL_MODEL)

  console.log('CACHEDIAG:', JSON.stringify(r))
  if (r.fetchStatus !== 200) throw new Error(`fetch status ${r.fetchStatus}`)
  if (!r.cacheHit || r.cacheBytes !== r.bytes) throw new Error('cache round-trip failed')
})
