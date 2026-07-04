# FaceStudio

Average and morph human faces **entirely in your browser**. No uploads, no server, no accounts — your photos never leave your device.

**→ [Live app](https://georgegach.github.io/face-average/)**

FaceStudio is a ground-up rewrite of the original `face-average` Python/dlib CLI as a WASM-powered static site. It uses Google's **MediaPipe Face Landmarker** (478-point dense mesh) for detection and a **WebGL2 piecewise-affine warp** for real-time morphing, all client-side.

## Features

- **Average** any number of faces into one, with per-face weight sliders, template-shape mode, Lab-style colour normalisation, and background options.
- **Morph** between two people with a live blend slider; export the animation as **WebM/MP4** or **GIF**.
- **Enhance** results with on-device **Real-ESRGAN** upscalers (photo / anime / general) via ONNX Runtime Web (WebGPU when available).
- **Webcam capture**, batch drag-and-drop, and a preset gallery to try instantly.
- Installable **PWA**, offline-capable after first load.

## Architecture

- **Vite + React + TypeScript + Tailwind**.
- ML runs in **Web Workers**; warping runs on the **GPU** (WebGL2).
- Models are **self-hosted** (fetched at build time, SHA-checked) and cached by a service worker.
- Deployed to **GitHub Pages** by CI, which also runs a Playwright smoke test that exercises the real WASM pipeline.

See [`PLAN.md`](./PLAN.md) for the full design.

## Development

```bash
npm ci
bash scripts/fetch-models.sh   # downloads MediaPipe + ONNX assets into public/models
npm run dev
```

## Legacy

The original Python averaging CLI (dlib, 68 landmarks, `.ff` cache format) lives on the [`legacy`](https://github.com/georgegach/face-average/tree/legacy) branch.

## Acknowledgements

- Original approach based on Satya Mallick's [Average Face tutorial](https://www.learnopencv.com/average-face-opencv-c-python-tutorial/).
- Landmarks by [MediaPipe](https://developers.google.com/mediapipe); upscaling by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

## License

MIT — see [LICENSE](./LICENSE).
