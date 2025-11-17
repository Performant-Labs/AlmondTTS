#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
DMG_PATH="$DIST_DIR/AlmondTTS.dmg"
CLI_TARGET="$DIST_DIR/almond_tts"
SPEC_FILE="$PROJECT_ROOT/almond_tts.spec"
PREP_SCRIPT="$PROJECT_ROOT/packaging/prepare_bundle_resources.sh"
FORMAT_SCRIPT="$PROJECT_ROOT/scripts/format_docs.py"
SIGN_IDENTITY="${ALMOND_TTS_CODESIGN_IDENTITY:--}"
PYINSTALLER_BIN="${PYINSTALLER_BIN:-}"
DMG_STAGING_DIR="$DIST_DIR/AlmondTTS_dmg"
DMG_ROOT_DIR="$DMG_STAGING_DIR/AlmondTTS"
SAMPLE_INPUT="$PROJECT_ROOT/input/0000-BasicGrammar.txt"
START_HERE_DOC="$PROJECT_ROOT/START_HERE.txt"
README_DOC="$PROJECT_ROOT/README.md"

usage() {
  cat <<EOF
Usage: ${0##*/} [--no-codesign] [--no-dmg]

Environment overrides:
  ALMOND_TTS_CODESIGN_IDENTITY  Codesign identity (defaults to ad-hoc "-")
EOF
}

NO_CODESIGN=0
NO_DMG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-codesign)
      NO_CODESIGN=1
      shift
      ;;
    --no-dmg)
      NO_DMG=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PYINSTALLER_BIN" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/pyinstaller" ]]; then
    PYINSTALLER_BIN="${VIRTUAL_ENV}/bin/pyinstaller"
  elif [[ -x "$PROJECT_ROOT/venv/bin/pyinstaller" ]]; then
    PYINSTALLER_BIN="$PROJECT_ROOT/venv/bin/pyinstaller"
  else
    PYINSTALLER_BIN="pyinstaller"
  fi
fi

echo "==> Staging resources"
"$PREP_SCRIPT"

if [[ -f "$FORMAT_SCRIPT" ]]; then
  echo "==> Formatting documentation"
  python3 "$FORMAT_SCRIPT"
fi

echo "==> Building via PyInstaller"
pushd "$PROJECT_ROOT" >/dev/null
"$PYINSTALLER_BIN" "$SPEC_FILE"
popd >/dev/null

CLI_BIN=""
CLI_PAYLOAD_DIR=""
if [[ -d "$CLI_TARGET" && -x "$CLI_TARGET/almond_tts" ]]; then
  CLI_BIN="$CLI_TARGET/almond_tts"
  CLI_PAYLOAD_DIR="$CLI_TARGET"
elif [[ -x "$CLI_TARGET" ]]; then
  CLI_BIN="$CLI_TARGET"
else
  echo "ERROR: CLI binary not found (looked for $CLI_TARGET/almond_tts and $CLI_TARGET)"
  exit 1
fi

if [[ $NO_CODESIGN -eq 0 ]]; then
  echo "==> Codesigning CLI binary with identity: $SIGN_IDENTITY"
  codesign --force --sign "$SIGN_IDENTITY" "$CLI_BIN"
else
  echo "==> Skipping codesign (--no-codesign)"
fi

if [[ $NO_DMG -eq 0 ]]; then
  echo "==> Preparing DMG staging directory"
  rm -rf "$DMG_STAGING_DIR"
  mkdir -p "$DMG_ROOT_DIR"
  if [[ -n "$CLI_PAYLOAD_DIR" ]]; then
    cp -R "$CLI_PAYLOAD_DIR"/. "$DMG_ROOT_DIR"
  else
    mkdir -p "$DMG_ROOT_DIR"
    cp "$CLI_BIN" "$DMG_ROOT_DIR/almond_tts"
  fi
  if [[ -f "$START_HERE_DOC" ]]; then
    echo "==> Copying START_HERE guide"
    cp "$START_HERE_DOC" "$DMG_ROOT_DIR/START_HERE.txt"
  fi
  if [[ -f "$README_DOC" ]]; then
    echo "==> Copying README"
    cp "$README_DOC" "$DMG_ROOT_DIR/README.md"
  fi
  if [[ -f "$SAMPLE_INPUT" ]]; then
    echo "==> Copying sample input file"
    mkdir -p "$DMG_ROOT_DIR/Samples"
    cp "$SAMPLE_INPUT" "$DMG_ROOT_DIR/Samples/0000-BasicGrammar.txt"
  fi
  echo "==> Creating DMG"
  rm -f "$DMG_PATH"
  hdiutil create -fs HFS+ -volname AlmondTTS \
    -srcfolder "$DMG_STAGING_DIR" "$DMG_PATH"
else
  echo "==> Skipping DMG creation (--no-dmg)"
fi

echo "==> Build complete"
echo "CLI binary: $CLI_BIN"
if [[ $NO_DMG -eq 0 ]]; then
  echo "DMG: $DMG_PATH"
fi
