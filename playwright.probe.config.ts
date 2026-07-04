import { defineConfig, devices } from '@playwright/test'

// Separate config for the heavy Enhance/upscale probe — long timeout, run via
// the probe workflow only (not part of the deploy smoke tests).
export default defineConfig({
  testDir: './probe',
  timeout: 360_000,
  expect: { timeout: 300_000 },
  fullyParallel: false,
  retries: 0,
  reporter: 'line',
  use: {
    baseURL: 'http://localhost:4173/face-average/',
    trace: 'off',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'npm run preview',
    url: 'http://localhost:4173/face-average/',
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
  },
})
