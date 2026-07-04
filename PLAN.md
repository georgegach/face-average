# FaceStudio — Implementation Plan

Rewrite of `face-average` (2018 Python/dlib CLI) as a fully client-side, WASM-powered static web app deployed to GitHub Pages. This document is the single source of truth for the implementing agent. Follow it phase by phase; each phase ends with a working, deployable state.

---

## 0. Decisions already made (do not re-litigate)

| Topic | Decision |
|---|---|
| ML engines | **Hybrid**: MediaPipe Tasks Face Landmarker (WASM/WebGL, 478 landmarks) for detection/landmarks + **ONNX Runtime Web** (WebGPU with WASM fallback) for face parsing and upscalers |
| Frontend | **Vite + React 18 + TypeScript + Tailwind CSS** |
| Repo | Create branch `legacy` from current `master` (preserves Python tool), then **full replace of `master`** with the new app at repo root. Deploy via GitHub Actions to GitHub Pages |
| Models | **Self-hosted**, served same-origin from the Pages bundle, cached by a service worker (offline-capable PWA) |
| Algorithm | **Modernized**: 478-pt dense piecewise-affine warping, Lab-space accumulation, parsing-mask-aware compositing, color normalization — not a faithful 68-pt port |
| Features | Multi-face averaging, two-person weighted blend/morph, template mode + preset gallery, webcam capture, N-way weight sliders, animated morph export (WebM/GIF), face-parsing compositing, manual landmark editor, AI upscalers (photo / anime / general variants) |
| Philosophy | YAGNI. No backend, no accounts, no analytics, no state libraries beyond what's listed. Everything runs in the browser. |

Naming: the app is called **FaceStudio** (site title: "FaceStudio — face averaging & morphing in your browser"). Repo name stays `face-average`.

---

## 1. Repo transition (Phase 0)

```bash
git checkout master
git branch legacy            # freeze the Python tool
git push origin legacy
# then on master: remove run.py, src/, datasets/ (keep a few preset images — see §7), results/
```

- New README.md: hero screenshot, link to live site `https://georgegach.github.io/face-average/`, one paragraph on architecture, note that the original Python CLI lives on the `legacy` branch.
- Keep `LICENSE`. Add `.nvmrc` (Node 20), `.gitignore` for node/vite.
- Commit messages: plain, no AI attribution (user rule).
- **Never run the app locally** (user rule): verify via `npm run build`? No — even that counts as local execution is ambiguous; building with `npm run build` is required by CI only. All verification happens through GitHub Actions logs (`gh run watch`). Write code, push, watch CI. Add a CI job that runs `tsc --noEmit`, `vite build`, and Playwright smoke tests **in CI** so correctness is verifiable without local runs.

## 2. Architecture overview

```
face-average/
├─ .github/workflows/deploy.yml     # build + test + deploy to Pages
├─ index.html
├─ vite.config.ts                   # base: '/face-average/'
├─ tailwind.config.ts
├─ public/
│  ├─ models/                       # self-hosted model files (see §3)
│  ├─ presets/                      # preset face galleries (see §7)
│  └─ icons/ manifest.webmanifest
├─ src/
│  ├─ app/                          # React shell, routing (single page, tab views)
│  ├─ ui/                           # dumb components (Slider, Dropzone, CanvasStage…)
│  ├─ state/                        # zustand store (single small store)
│  ├─ engine/
│  │  ├─ landmarks.ts               # MediaPipe FaceLandmarker wrapper (worker)
│  │  ├─ parsing.ts                 # ONNX face-parsing wrapper (worker)
│  │  ├─ upscale.ts                 # ONNX Real-ESRGAN wrappers (worker, tiled)
│  │  ├─ warp.ts                    # piecewise-affine warp (triangulated mesh, WebGL2)
│  │  ├─ average.ts                 # averaging/blending pipeline
│  │  ├─ color.ts                   # Lab conversion, color normalization
│  │  ├─ morph.ts                   # A↔B morph frame generator
│  │  └─ export.ts                  # PNG/WebM/GIF export
│  └─ workers/                      # actual Worker entry points
└─ e2e/                             # Playwright smoke tests (run in CI)
```

Key principles:
- **All ML and pixel work happens in Web Workers** (`landmarks.worker.ts`, `onnx.worker.ts`); the main thread only orchestrates and draws final results. Communicate with transferable `ImageBitmap`/`ArrayBuffer`s.
- **Warping runs on WebGL2** in the main thread (or OffscreenCanvas in a worker where supported): upload the 478-pt source/destination meshes as vertex attributes, let the GPU do piecewise-affine interpolation per triangle. This is the single biggest perf win vs. the old CPU Delaunay loop and is ~50 lines of shader.
- State: one `zustand` store: `faces: Face[]`, `mode`, `settings`, `result`. A `Face` = `{ id, bitmap, landmarks: Float32Array(478*2), weight, enabled, maskBitmap? }`.
- COOP/COEP headers are NOT available on GitHub Pages → **do not** rely on multi-threaded WASM (SharedArrayBuffer). Use single-threaded WASM + SIMD builds of onnxruntime-web and MediaPipe; prefer WebGPU EP when available, `wasm` EP otherwise.

## 3. Models (self-hosted in `public/models/`)

Do **not** commit model binaries to git. Instead, fetch them at build time in CI (curl into `public/models/` before `vite build`) with pinned URLs + SHA-256 checks recorded in `scripts/fetch-models.sh`. A `models.json` manifest (name, file, bytes, sha256, license) ships with the site and drives the service-worker precache and the in-app download UI.

| Model | Source | Purpose | ~Size |
|---|---|---|---|
| `face_landmarker.task` | MediaPipe (Google storage, pinned version) | 478 landmarks + blendshapes + face transform | ~3.7 MB |
| MediaPipe tasks-vision WASM | npm `@mediapipe/tasks-vision` (copy wasm assets to public) | runtime | ~6 MB |
| `face_parsing_bisenet.onnx` (BiSeNet, CelebAMask-HQ, 512×512, 19 classes) | onnx export (widely mirrored; pin a known-good one and verify hash) | hair/skin/bg masks for seam-aware compositing | ~13 MB fp16 |
| `realesrgan-x4-photo.onnx` (RealESRGAN_x4plus, fp16) | Real-ESRGAN releases → onnx | photo upscaling | ~17 MB |
| `realesrgan-x4-anime.onnx` (x4plus-anime-6B) | same | anime/illustration upscaling | ~4.5 MB |
| `realesr-general-x4v3.onnx` | same | general/nature, denoising variant | ~5 MB |
| onnxruntime-web wasm/webgpu assets | npm `onnxruntime-web` | runtime | ~10 MB (lazy) |

Loading policy (YAGNI, lazy):
- Landmarker loads eagerly on first image drop (it's needed for everything).
- Parsing model loads lazily the first time "smart compositing" is enabled (on by default for Average mode, so effectively on first average run — show a one-time progress toast).
- Upscalers load only when the user opens the Enhance panel and picks a variant.
- Service worker (Workbox via `vite-plugin-pwa`) precaches app shell + landmarker; runtime-caches ONNX models with cache-first strategy so second visits are offline-capable.

If a pinned ONNX model URL proves unreliable in CI, fall back to committing it via Git LFS — but try the fetch-at-build approach first.

## 4. Core pipeline (engine)

### 4.1 Landmarking
- `FaceLandmarker` from `@mediapipe/tasks-vision`, `runningMode: 'IMAGE'` for uploads, `'VIDEO'` for webcam.
- Normalize output to pixel coords `Float32Array(478*2)`. Keep MediaPipe's canonical triangulation (`FACEMESH_TESSELATION` → precompute a triangle index list of the 478-pt mesh, ship as a constant `TRIANGULATION: Uint16Array`). **Augment with 8 boundary points** (corners + edge midpoints of the output canvas) and ~16 interpolated hairline points (extrapolated above the forehead from mesh geometry) so warps cover the full frame like the old boundaryPts did, but smoother. Triangulate the augmented set once with a small Delaunay lib (`delaunator`) over the *destination* mesh; reuse indices for all faces.
- Multi-face images: run with `numFaces: 4`; if >1 face found, show a face-picker overlay (tap the face to use).

### 4.2 Normalization & averaging (Average mode)
1. Choose output size (default 1024×1365, user-adjustable; keep the old 3:4 spirit).
2. Similarity-align every face by eye centers (landmarks 468/473 — iris centers, far more stable than dlib 36/45) to canonical positions: eye line at `h/2.5`, inter-eye distance `0.3*w` (port the old params, expose as advanced sliders).
3. Destination mesh = weighted average of aligned landmark sets (weights from per-face sliders, default uniform). Template mode: destination mesh = the template face's aligned mesh instead.
4. GPU piecewise-affine warp each aligned image to the destination mesh.
5. Accumulate in **linear-light float RGBA**, then optional **color normalization**: per-face mean/std transfer in Lab toward the group mean before accumulation (toggle, on by default) — kills the sallow color-cast problem.
6. **Seam-aware compositing** (toggle, on by default): run face parsing per input, build a soft face+hair mask, warp masks with the images, and use per-pixel weighted accumulation `Σ(wᵢ·maskᵢ·imgᵢ)/Σ(wᵢ·maskᵢ)`, falling back to plain average where mask coverage is zero. Background: fill with blurred average or a solid studio color (user picks).

### 4.3 Blend/Morph mode (two faces)
- Slider `t ∈ [0,1]`: destination mesh = lerp(meshA, meshB, t); result = cross-dissolve of both warped images with weight `(1−t, t)`. Live-updating as the slider moves (GPU warp makes this real-time).
- "Animate" button: render N frames (default 48) across an ease-in-out curve, encode to **WebM** via `MediaRecorder` on a canvas stream (universal) and **GIF** via `gifenc` (small, fast). Boomerang toggle.

### 4.4 Landmark editor
- Zoomable/pannable canvas overlay (single component, pointer events, no library) showing the 478 points at low opacity with the ~30 semantically important ones (eyes, brows, nose, mouth, jawline — define an index list) rendered larger and draggable. Dragging a key point moves its neighbors with Gaussian falloff over mesh-adjacency distance (radius slider). "Reset points" per face. Edits invalidate cached warps.

### 4.5 Enhance (upscalers)
- Panel on the result: variant picker (Photo / Anime / General-Nature), 2× or 4× (2× = 4× then downscale). Run tiled (256px tiles, 16px overlap, feathered blend) in the ONNX worker to bound memory; progress bar per tile. WebGPU EP when `navigator.gpu` exists, else wasm+SIMD with a "this may take a while" note.

### 4.6 Webcam
- `getUserMedia` capture view with live landmark overlay (VIDEO mode) as framing feedback; shutter button freezes a frame into the face tray like any upload. (Live continuous morphing is out of scope — YAGNI; the capture path gives 90% of the value.)

## 5. UI/UX spec

Single-page app, dark studio aesthetic (near-black `#0b0d10` bg, subtle glass panels, one accent color — electric cyan; Tailwind, Inter font, generous whitespace, tasteful motion via CSS transitions only).

Layout (desktop):
- **Left rail — Face Tray**: drag-drop zone ("Drop faces or click / press W for webcam"), thumbnail cards with detected-landmark badge, per-face weight slider (Average mode), enable toggle, template-star button (Template mode), delete. Batch drop of dozens of files with a queue + progress.
- **Center — Stage**: large canvas with the result; before/after scrub divider (drag a vertical handle comparing raw-average vs enhanced/composited); zoom/pan.
- **Right rail — Controls**, tabbed by mode: **Average · Morph · Enhance**. Mode-specific controls per §4, plus output size, background style, advanced accordion (eye distance/height, color-normalize toggle, mask toggle).
- **Top bar**: logo, mode tabs, preset gallery button, Export button (PNG @1×/2×, WebM/GIF in Morph mode), GitHub link.
- Mobile: rails collapse into bottom sheets; everything must be usable at 390px width.

UX details that matter:
- First visit: empty state shows the preset gallery ("Try: US Presidents") so users see the magic in one click with zero uploads.
- All heavy ops show determinate progress (model download %, per-image landmarking i/N, upscale tiles).
- Face with no detection: card shows warning state + "open landmark editor to place manually" is **not** offered (YAGNI — manual placement of 478 pts is absurd); instead offer "retry" and explain the photo needs a clear frontal face.
- Everything keyboard accessible; canvas results get alt text; respect `prefers-reduced-motion`.
- No data ever leaves the browser — state this prominently in the footer ("100% local. Your photos never leave your device.").

## 6. Deployment (GitHub Actions → Pages)

`.github/workflows/deploy.yml`:
1. `npm ci`
2. `scripts/fetch-models.sh` (curl + sha256 verify into `public/models/`)
3. `tsc --noEmit && npm run build` (vite `base: '/face-average/'`)
4. Playwright smoke test against `vite preview`: app loads, drop two preset images via fixture, average renders non-blank canvas (read pixels, assert variance > threshold), morph slider changes output. Use `--project=chromium` only. This is the substitute for local verification — it must actually exercise the WASM pipeline.
5. Deploy with `actions/upload-pages-artifact` + `actions/deploy-pages`.
Note: this repo's other projects deploy to an Oracle VM; **this one is an exception** — it deploys to GitHub Pages (static site, no server). Enable Pages "GitHub Actions" source in repo settings (ask user if permissions block `gh api` for that).

## 7. Presets

Keep 6 US presidents images from `datasets/us-mp/president/` → `public/presets/presidents/` (resized to max 800px, stripped EXIF). Delete `.ff` files (obsolete). Preset manifest JSON with attribution. (GoT images are not in the repo; skip.)

## 8. Phases & acceptance criteria

Each phase = one or more commits pushed to master, verified green in CI before starting the next.

- **P0 Repo transition**: `legacy` branch pushed; master cleaned; Vite+React+TS+Tailwind scaffold with dark shell, empty tray, CI deploying a placeholder page to Pages successfully. ✅ Site live.
- **P1 Landmarks + Average**: MediaPipe in worker, tray with drop/thumbnails, GPU warp, plain average (no masks/color-norm), preset gallery works end-to-end. ✅ Presidents average visibly comparable-or-better than the legacy result. Playwright test asserts it.
- **P2 Quality**: Lab color normalization, BiSeNet parsing masks + seam-aware compositing, background options, weight sliders, template mode. ✅ No hair-halo artifact on presets.
- **P3 Morph**: two-face blend slider (real-time), WebM+GIF export, boomerang. ✅ 48-frame WebM exports in CI test (assert non-zero blob).
- **P4 Editor + Webcam**: landmark editor with falloff dragging; webcam capture. ✅ Manual nudge visibly changes result.
- **P5 Enhance + PWA**: three ONNX upscalers, tiled, WebGPU/wasm fallback; service-worker precache/offline; manifest + icons. ✅ Lighthouse PWA installable; 4× upscale completes on a 1024px result.
- **P6 Polish**: mobile sheets, keyboard/a11y pass, README with screenshots/GIF, footer privacy note.

## 9. Explicit non-goals (YAGNI)

No backend, accounts, sharing links, project-file format, live continuous webcam morphing, StyleGAN/diffusion blending, i18n, telemetry, IndexedDB persistence of user photos (session-only), and no support for browsers without WASM SIMD (show a friendly unsupported notice).

## 10. Risks & fallbacks

- **ONNX model sourcing**: if a pinned BiSeNet/Real-ESRGAN ONNX URL is flaky, vendor via Git LFS. If BiSeNet quality disappoints, fallback mask = convex hull of the 478 mesh feathered outward (still better than nothing; ship behind the same toggle).
- **WebGPU absent** (Safari < 18, Firefox): wasm EP works but slow for upscalers → cap wasm-EP upscales at 2× with a notice.
- **Memory** on mobile with 30+ faces: downscale inputs to max 1600px on ingest; process accumulation incrementally (running sum), never hold all warped frames.
- **MediaRecorder WebM on Safari**: falls back to MP4 (`video/mp4` mimeType) — detect via `MediaRecorder.isTypeSupported`.
