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

# ---- Face landmarker (REQUIRED) -------------------------------------------
fetch \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
  "$OUT/face_landmarker.task" "-" "required" || exit 1

# ---- Optional ONNX models (lazy features: parsing + upscalers) ------------
# Pinned mirrors; if any is unreachable the related in-app feature disables
# gracefully. Replace URLs / add sha256 pins as better-hosted exports appear.
fetch \
  "https://huggingface.co/onnx-community/face-parsing-bisenet/resolve/main/model.onnx" \
  "$OUT/face_parsing_bisenet.onnx" "-" "optional"

fetch \
  "https://huggingface.co/Xenova/real-esrgan-x4plus/resolve/main/onnx/model.onnx" \
  "$OUT/realesrgan-x4-photo.onnx" "-" "optional"

fetch \
  "https://huggingface.co/Xenova/real-esrgan-x4plus-anime/resolve/main/onnx/model.onnx" \
  "$OUT/realesrgan-x4-anime.onnx" "-" "optional"

echo "== model manifest =="
ls -la "$OUT"
