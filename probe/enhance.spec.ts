import { test, expect, type Page } from '@playwright/test'

const canvasW = (page: Page) =>
  page.getByTestId('result-canvas').evaluate((c: HTMLCanvasElement) => c.width)

test('upscale 4x runs end-to-end (wasm EP)', async ({ page }) => {
  const logs: string[] = []
  page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`))
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`))
  page.on('requestfailed', (r) => logs.push(`[reqfail] ${r.url()} ${r.failure()?.errorText}`))

  await page.goto('/')
  await page.getByRole('button', { name: /US Presidents/ }).click()
  await expect(page.getByText('478 pts').nth(3)).toBeVisible({ timeout: 90_000 })
  await page.getByRole('button', { name: /Average \d+ faces/ }).click()
  await expect(page.getByTestId('result-canvas')).toBeVisible({ timeout: 60_000 })
  const before = await canvasW(page)
  logs.push(`original canvas width = ${before}`)

  await page.getByRole('button', { name: 'Enhance', exact: true }).click()
  await page.getByRole('button', { name: 'General / Nature' }).click()
  await page.getByRole('button', { name: /Upscale 4/ }).click()

  let failure: unknown = null
  try {
    await expect
      .poll(() => canvasW(page), { timeout: 300_000, intervals: [3000] })
      .toBeGreaterThan(before)
  } catch (e) {
    failure = e
  }

  const statusMsg = await page
    .locator('p')
    .filter({ hasText: /Upscal|failed|reach|model/ })
    .allInnerTexts()
    .catch(() => [])

  console.log('\n================ BROWSER LOG ================')
  for (const l of logs) console.log(l)
  console.log('status messages on screen:', JSON.stringify(statusMsg))
  console.log('final canvas width:', await canvasW(page).catch(() => 'n/a'))
  console.log('============================================\n')

  if (failure) throw failure
})
