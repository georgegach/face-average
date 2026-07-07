# FaceStudio

A private, **on-device face editor** — retouch, restyle, reshape and re-age portraits **entirely in your browser**. No uploads, no server, no accounts — your photos never leave your device, and it's free.

**→ [Live app](https://georgegach.github.io/facestudio/)**

FaceStudio is a ground-up rewrite of the original `face-average` Python/dlib CLI as a WASM-powered static site. It uses Google's **MediaPipe Face Landmarker** (478-point dense mesh) for detection and a **WebGL2 piecewise-affine warp** for real-time face manipulation, all client-side.

## Features

The flagship is a full **face editor**; averaging, morphing, replace, enhance, and a "future baby" blend are auxiliary tools built on the same on-device engine.

- **Edit** — the flagship, FaceApp/Facetune-style but 100% on-device: skin smoothing, teeth whitening, makeup (lips / blush / brows / eye colour), hair recolouring, background bokeh or studio, and vignette via **BiSeNet face parsing**; shape tools (smile, eye size, nose/face slim, hair volume) as landmark warps; **re-aging** with a CI-converted **FRAN** U-Net (MIT) that edits as an identity-preserving delta; one-tap **Looks**, a **hold-to-compare** before/after, and **share**.
- **Future Baby** — for fun, blend two parents and de-age the result to glimpse a future child (composes the averaging + re-aging engines). Not a genetic prediction.
- **Average** any number of faces into one, with per-face weight sliders, template-shape mode, Lab-style colour normalisation, and background options.
- **Morph** between two people with a live blend slider; export the animation as **WebM/MP4** or **GIF**.
- **Replace** a face in any photo from a multi-image *pose bank* of one person — the closest-pose source is warped onto the target at native resolution, colour-matched and seam-feathered. No generative models, no quality loss.
- **Enhance** results with on-device **Real-ESRGAN** upscalers (photo / anime / general) via ONNX Runtime Web (WebGPU when available).
- **Webcam capture**, batch drag-and-drop, and a preset gallery to try instantly.
- Installable **PWA**, offline-capable after first load.

## Architecture

- **Vite + React + TypeScript + Tailwind**.
- ML runs on the **main thread** via WASM/WebGPU — MediaPipe's loader needs `importScripts`, unavailable in ES-module workers — while warping runs on the **GPU** (WebGL2).
- Models are **self-hosted** (fetched at build time, SHA-checked) and cached by a service worker.
- Deployed to **GitHub Pages** by CI, which runs `vitest` unit tests over the pure engine maths and a Playwright smoke test that exercises the real WASM pipeline before every deploy.

## Development

```bash
npm install
bash scripts/fetch-models.sh   # downloads MediaPipe + ONNX assets into public/models
npm run dev
npm test                       # unit tests (vitest); npm run test:e2e for the Playwright smoke suite
```

## Privacy

Every pixel operation runs **on your device** — photos are never uploaded, and there are no accounts. The app collects **anonymous usage analytics** (PostHog, EU region): page views and product events such as which tool was run, its duration and success, exports, and model downloads. **No images, filenames, or personal data are ever sent**, and analytics is disabled in development and automated (CI) runs.

## Legacy

The original Python averaging CLI (dlib, 68 landmarks, `.ff` cache format) lives on the [`legacy`](https://github.com/georgegach/facestudio/tree/legacy) branch.

## Acknowledgements

- Original approach based on Satya Mallick's [Average Face tutorial](https://www.learnopencv.com/average-face-opencv-c-python-tutorial/).
- Landmarks by [MediaPipe](https://developers.google.com/mediapipe); upscaling by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

## License

Copyright © 2018–2026 George Gach. Licensed under **GPL-3.0-only** — see [LICENSE](./LICENSE).
