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

# ---- Optional ONNX upscalers (lazy Enhance feature) -----------------------
# Standard 4x ESRGAN models (dynamic shape, NCHW float 0..1 in/out). If a
# mirror is unreachable the related variant disables gracefully in-app.
UP="https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main"
fetch "$UP/4x-UltraSharpV2_Lite.onnx" "$OUT/upscale-photo.onnx"   "-" "optional"
fetch "$UP/4x-ClearRealityV1.onnx"    "$OUT/upscale-general.onnx" "-" "optional"
fetch "$UP/4x-AnimeSharp.onnx"        "$OUT/upscale-anime.onnx"   "-" "optional"

echo "== model manifest =="
ls -la "$OUT" "$OUT/ort" 2>/dev/null
