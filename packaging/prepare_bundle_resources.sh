#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOURCES_DIR="$ROOT_DIR/bundle_resources"
MODELS_DIR="$RESOURCES_DIR/models"
MODEL_NAME="tts_models/multilingual/multi-dataset/xtts_v2"

mkdir -p "$MODELS_DIR"

dir_has_files() {
  local dir="$1"
  [[ -d "$dir" ]] && [[ -n "$(find "$dir" -mindepth 1 -print -quit)" ]]
}

copy_cache_if_present() {
  local source="$1"
  [[ -z "${source:-}" ]] && return 1
  if dir_has_files "$source"; then
    echo "Copying cached models from: $source"
    mkdir -p "$MODELS_DIR"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$source"/ "$MODELS_DIR"/
    else
      cp -R "$source"/. "$MODELS_DIR"/
    fi
    return 0
  fi
  return 1
}

echo "Preparing bundled resources in $RESOURCES_DIR"

if dir_has_files "$MODELS_DIR"; then
  echo "✓ Model cache already present. Nothing to do."
  exit 0
fi

echo "→ Searching for an existing Coqui cache to reuse..."
copy_cache_if_present "${ALMOND_TTS_MODEL_DIR:-}" || true
copy_cache_if_present "${COQUI_TTS_CACHE_DIR:-}" || true
copy_cache_if_present "$HOME/Projects/tts-models" || true
copy_cache_if_present "$HOME/Documents/AlmondTTS/models" || true
copy_cache_if_present "$HOME/.local/share/tts" || true

if dir_has_files "$MODELS_DIR"; then
  echo "✓ Copied cached models into bundle_resources/models"
  exit 0
fi

echo "→ No existing cache found. Downloading $MODEL_NAME (this can take several minutes)..."
COQUI_TTS_CACHE_DIR="$MODELS_DIR" python - <<'PY'
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

print(f"Downloading {MODEL_NAME} ...")
tts = TTS(model_name=MODEL_NAME)
tts.to("cpu")
print("✓ Download complete")
PY

echo "✓ Models downloaded to $MODELS_DIR"
