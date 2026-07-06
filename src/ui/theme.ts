export type Theme = 'light' | 'dark'

const KEY = 'facestudio-theme'

export function getTheme(): Theme {
  const stored = localStorage.getItem(KEY)
  return stored === 'dark' ? 'dark' : 'light' // light is the default
}

export function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle('dark', theme === 'dark')
  const meta = document.querySelector('meta[name="theme-color"]')
  if (meta) meta.setAttribute('content', theme === 'dark' ? '#0a0a0e' : '#f0f2f7')
}

export function setTheme(theme: Theme) {
  localStorage.setItem(KEY, theme)
  applyTheme(theme)
}

export function initTheme() {
  applyTheme(getTheme())
}
