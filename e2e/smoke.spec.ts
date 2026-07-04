import { test, expect, type Page } from '@playwright/test'

// Reads a canvas and returns the luminance variance of its pixels. A blank or
// solid canvas has ~0 variance; a real rendered face has a large value.
async function canvasVariance(page: Page, testid: string): Promise<number> {
  return page.evaluate((id) => {
    const c = document.querySelector(`[data-testid="${id}"]`) as HTMLCanvasElement
    if (!c || !c.width) return -1
    const ctx = c.getContext('2d')!
    const { data } = ctx.getImageData(0, 0, c.width, c.height)
    let sum = 0
    let sq = 0
    let n = 0
    for (let i = 0; i < data.length; i += 40) {
      const l = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
      sum += l
      sq += l * l
      n++
    }
    const mean = sum / n
    return sq / n - mean * mean
  }, testid)
}

test('averages preset faces into a real image', async ({ page }) => {
  page.on('console', (m) => console.log(`[browser:${m.type()}] ${m.text()}`))
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`))
  page.on('requestfailed', (r) =>
    console.log(`[reqfail] ${r.url()} — ${r.failure()?.errorText}`),
  )
  await page.goto('/')
  await expect(page.getByText('Blend faces into one')).toBeVisible()

  // Load the presidents preset from the empty-state gallery.
  await page.getByRole('button', { name: /US Presidents/ }).click()

  // Wait until at least 4 faces have been detected (478 pts badge).
  await expect(page.getByText('478 pts ✓').nth(3)).toBeVisible({ timeout: 90_000 })

  // Render the average.
  await page.getByRole('button', { name: /Average \d+ faces/ }).click()
  await expect(page.getByTestId('result-canvas')).toBeVisible({ timeout: 60_000 })

  // Result canvas must contain real, varied pixels — not a blank frame.
  await expect
    .poll(() => canvasVariance(page, 'result-canvas'), { timeout: 30_000 })
    .toBeGreaterThan(50)
})

test('morph mode renders and responds to the blend slider', async ({ page }) => {
  page.on('console', (m) => console.log(`[browser:${m.type()}] ${m.text()}`))
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`))
  await page.goto('/')
  await page.getByRole('button', { name: /US Presidents/ }).click()
  await expect(page.getByText('478 pts ✓').nth(1)).toBeVisible({ timeout: 90_000 })

  await page.getByRole('button', { name: 'Morph', exact: true }).click()
  await page.waitForTimeout(1000)
  console.log('[diag] body text after Morph click:\n' + (await page.locator('main').innerText()))
  await expect(page.getByTestId('morph-canvas')).toBeVisible({ timeout: 30_000 })
  await expect
    .poll(() => canvasVariance(page, 'morph-canvas'), { timeout: 30_000 })
    .toBeGreaterThan(50)

  const before = await canvasVariance(page, 'morph-canvas')
  const slider = page.locator('input[type="range"]').last()
  await slider.fill('0.05')
  await page.waitForTimeout(500)
  const after = await canvasVariance(page, 'morph-canvas')
  expect(Math.abs(after - before)).toBeGreaterThan(0.0001)
})
