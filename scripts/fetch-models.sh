#!/usr/bin/env bash
# Fetches self-hosted model assets into public/models/ at build time.
# Required models fail the build if missing; optional (lazy-loaded) models are
# best-effort so the core averaging pipeline always ships even if a mirror is down.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/public/models"
mkdir -p "$OUT"

# ---- helpers ---------------------------------------------------------------
fetch() {
  # fetch <url> <dest> <sha256|-> <required|optional>
  local url="$1" dest="$2" sha="$3" req="$4"
  if [ -f "$dest" ]; then echo "cached: $(basename "$dest")"; return 0; fi
  echo "fetch : $(basename "$dest")  <- $url"
  if ! curl -fsSL --retry 3 --retry-delay 2 -o "$dest.part" "$url"; then
    echo "  !! download failed"
    rm -f "$dest.part"
    [ "$req" = "required" ] && return 1 || { echo "  (optional, skipping)"; return 0; }
  fi
  if [ "$sha" != "-" ]; then
    local got
    got="$(shasum -a 256 "$dest.part" | awk '{print $1}')"
    if [ "$got" != "$sha" ]; then
      echo "  !! sha256 mismatch: got $got"
      rm -f "$dest.part"
      [ "$req" = "required" ] && return 1 || return 0
    fi
  fi
  mv "$dest.part" "$dest"
}

# ---- MediaPipe runtime (wasm) ---------------------------------------------
# Copy the SIMD wasm assets shipped inside the npm package (same-origin hosting).
MP_WASM_SRC="$ROOT/node_modules/@mediapipe/tasks-vision/wasm"
if [ -d "$MP_WASM_SRC" ]; then
  mkdir -p "$OUT/wasm"
  cp -f "$MP_WASM_SRC"/* "$OUT/wasm/" 2>/dev/null || true
  echo "copied: mediapipe wasm assets"
else
  echo "!! @mediapipe/tasks-vision wasm not found — run npm ci first" >&2
  exit 1
fi

# ---- onnxruntime-web runtime (wasm/mjs) -----------------------------------
# Self-host so ort loads its wasm same-origin (set via env.wasm.wasmPaths).
ORT_SRC="$ROOT/node_modules/onnxruntime-web/dist"
if [ -d "$ORT_SRC" ]; then
  mkdir -p "$OUT/ort"
  cp -f "$ORT_SRC"/*.wasm "$OUT/ort/" 2>/dev/null || true
  cp -f "$ORT_SRC"/*.mjs "$OUT/ort/" 2>/dev/null || true
  echo "copied: onnxruntime-web runtime assets"
else
  echo "!! onnxruntime-web dist not found — run npm ci first" >&2
  exit 1
fi

# ---- Face landmarker (REQUIRED) -------------------------------------------
fetch \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
  "$OUT/face_landmarker.task" "-" "required" || exit 1

# ---- Face parsing (BiSeNet ResNet18, MIT — yakhyo/face-parsing) (REQUIRED) --
# 19-class CelebAMask-HQ segmentation at 512x512; powers the Edit mode's
# retouch/makeup/hair/background tools.
fetch \
  "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx" \
  "$OUT/face_parsing.onnx" "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f" "required" || exit 1

# ---- Face re-aging (FRAN U-Net, MIT — CI-converted, see convert-fran.yml) --
# Optional: the asset exists only after the one-time conversion workflow has run;
# builds stay green without it and the Age tool reports itself unavailable.
fetch \
  "https://github.com/georgegach/face-average/releases/download/models/fran.onnx" \
  "$OUT/fran.onnx" "-" "optional"

# Note: ONNX upscalers are NOT bundled — they exceed GitHub Pages' deploy size
# and instead load at runtime from HuggingFace (see src/engine/models.ts),
# cached by the service worker.

echo "== model manifest =="
ls -la "$OUT" "$OUT/ort" 2>/dev/null
