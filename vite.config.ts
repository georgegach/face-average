import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

// GitHub Pages serves this project under /facestudio/
export default defineConfig({
  base: '/facestudio/',
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.svg', 'presets/**/*'],
      manifest: {
        name: 'FaceStudio',
        short_name: 'FaceStudio',
        description:
          'Private on-device face editor — retouch, restyle, reshape and re-age portraits entirely in your browser.',
        theme_color: '#0a0a0e',
        background_color: '#0a0a0e',
        display: 'standalone',
        start_url: '/facestudio/',
        scope: '/facestudio/',
        icons: [
          { src: 'favicon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any maskable' },
        ],
      },
      workbox: {
        // Precache only the small app shell; large wasm + models are
        // runtime-cached on first use (cache-first) for offline capability.
        maximumFileSizeToCacheInBytes: 4 * 1024 * 1024,
        globPatterns: ['**/*.{js,css,html}'],
        globIgnores: ['**/models/**', '**/*.wasm'],
        runtimeCaching: [
          {
            urlPattern: ({ url }) =>
              url.pathname.includes('/models/') || url.pathname.endsWith('.wasm'),
            handler: 'CacheFirst',
            options: {
              cacheName: 'facestudio-models',
              expiration: { maxEntries: 30, maxAgeSeconds: 60 * 60 * 24 * 90 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
          // NOTE: HuggingFace upscaler downloads are deliberately NOT routed
          // through Workbox — it mishandles HF's 302 redirect and returns 405.
          // Those models are cached in app code via the Cache API instead
          // (see engine/download.ts:fetchWithProgressCached).
        ],
      },
    }),
  ],
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2022',
  },
})
