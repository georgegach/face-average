import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Semantic tokens backed by CSS variables so the same utility classes
        // resolve to light or dark values depending on the active theme.
        bg: 'rgb(var(--bg) / <alpha-value>)',
        surface: 'rgb(var(--surface) / <alpha-value>)',
        surface2: 'rgb(var(--surface-2) / <alpha-value>)',
        surface3: 'rgb(var(--surface-3) / <alpha-value>)',
        edge: 'rgb(var(--edge) / <alpha-value>)',
        content: 'rgb(var(--content) / <alpha-value>)',
        muted: 'rgb(var(--muted) / <alpha-value>)',
        faint: 'rgb(var(--faint) / <alpha-value>)',
        accent: {
          DEFAULT: 'rgb(var(--accent) / <alpha-value>)',
          soft: 'rgb(var(--accent-soft) / <alpha-value>)',
          ink: 'rgb(var(--accent-ink) / <alpha-value>)',
          hi: 'rgb(var(--accent-hi) / <alpha-value>)',
        },
      },
      fontFamily: {
        // SF Pro on Apple devices, best-native elsewhere. No webfont download —
        // keeps the "nothing leaves your device" promise honest.
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          'SF Pro Text',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'system-ui',
          'sans-serif',
        ],
      },
      boxShadow: {
        glass: '0 18px 44px rgb(var(--shadow) / 0.25)',
      },
    },
  },
  plugins: [],
} satisfies Config
