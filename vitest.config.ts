import { defineConfig } from 'vitest/config'

// Unit tests target the pure, DOM-free engine math (geometry, pose, colour,
// masks, schedules). The browser/GPU/ORT paths are covered by the Playwright
// smoke suite instead, so the test bootstrap needs no Vite plugins.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/engine/__tests__/**/*.test.ts'],
  },
})
