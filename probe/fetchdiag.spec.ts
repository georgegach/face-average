import { test } from '@playwright/test'

const URL_MODEL =
  'https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x-ClearRealityV1.onnx'

test('diagnose model fetch', async ({ page }) => {
  await page.goto('/')

  const raw = await page.evaluate(async (u) => {
    const out: Record<string, unknown> = {}
    out.swController = !!navigator.serviceWorker?.controller
    try {
      const res = await fetch(u)
      out.plainStatus = res.status
      out.plainType = res.type
      out.plainRedirected = res.redirected
      out.finalUrl = res.url
      out.contentLength = res.headers.get('content-length')
    } catch (e) {
      out.plainErr = String(e)
    }
    return out
  }, URL_MODEL)
  console.log('DIAG raw (SW may be uncontrolled on first load):', JSON.stringify(raw))

  // Reload so the service worker controls the page, then retry.
  await page.reload()
  await page.waitForTimeout(1500)
  const viaSW = await page.evaluate(async (u) => {
    const out: Record<string, unknown> = {}
    out.swController = !!navigator.serviceWorker?.controller
    try {
      const res = await fetch(u)
      out.status = res.status
      out.type = res.type
      out.redirected = res.redirected
      out.finalUrl = res.url
    } catch (e) {
      out.err = String(e)
    }
    return out
  }, URL_MODEL)
  console.log('DIAG viaSW:', JSON.stringify(viaSW))
})
