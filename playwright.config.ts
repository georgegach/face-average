import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  retries: 0,
  reporter: 'line',
  use: {
    baseURL: 'http://localhost:4173/facestudio/',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'npm run preview',
    url: 'http://localhost:4173/facestudio/',
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
  },
})
