import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

// GitHub Pages serves this project under /face-average/
export default defineConfig({
  base: '/face-average/',
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.svg', 'presets/**/*'],
      manifest: {
        name: 'FaceStudio',
        short_name: 'FaceStudio',
        description: 'Face averaging & morphing in your browser',
        theme_color: '#0b0d10',
        background_color: '#0b0d10',
        display: 'standalone',
        start_url: '/face-average/',
        scope: '/face-average/',
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
