import posthog from 'posthog-js'

// Anonymous product analytics on the shared PostHog project. The project key is
// a public, write-only ingestion token (safe to ship in a static site). We only
// ever send feature-usage events — never images, filenames, or landmark data,
// so the app's on-device privacy promise holds.
//
// Disabled in dev and under automation (Playwright sets navigator.webdriver) so
// only real end-user usage reaches the dataset.
const POSTHOG_KEY = 'phc_oSRe2ukH7B7yRszYXmcoFPRXQWRd3WFhEp5CFzYyhruF'
const POSTHOG_HOST = 'https://eu.i.posthog.com'

const enabled =
  import.meta.env.PROD && typeof navigator !== 'undefined' && !navigator.webdriver

export function initAnalytics(): void {
  if (!enabled) return
  posthog.init(POSTHOG_KEY, {
    api_host: POSTHOG_HOST,
    autocapture: false, // explicit events only — no DOM scraping
    capture_pageview: true,
    disable_session_recording: true,
    persistence: 'localStorage', // no cookies
    person_profiles: 'identified_only', // stay anonymous; we never identify
  })
}

export function capture(event: string, props?: Record<string, unknown>): void {
  if (enabled) posthog.capture(event, props)
}

/**
 * Time a pipeline run and report its outcome, then rethrow so callers keep their
 * own error handling. Reports the mode, caller-supplied context (counts, model
 * kind — never image data), success flag and wall-clock duration.
 */
export async function tracked<T>(
  mode: string,
  props: Record<string, unknown>,
  fn: () => Promise<T>,
): Promise<T> {
  const t0 = performance.now()
  try {
    const r = await fn()
    capture('pipeline_run', {
      mode,
      ...props,
      success: true,
      duration_ms: Math.round(performance.now() - t0),
    })
    return r
  } catch (e) {
    capture('pipeline_run', {
      mode,
      ...props,
      success: false,
      duration_ms: Math.round(performance.now() - t0),
      error: (e as Error).message,
    })
    throw e
  }
}
