#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
APP_PATH="$DIST_DIR/AlmondTTS.app"
DMG_PATH="$DIST_DIR/AlmondTTS.dmg"
CLI_TARGET="$DIST_DIR/almond_tts"
SPEC_FILE="$PROJECT_ROOT/almond_tts.spec"
PREP_SCRIPT="$PROJECT_ROOT/packaging/prepare_bundle_resources.sh"
ICON_PATH="${ALMOND_TTS_ICON_PATH:-}"
SIGN_IDENTITY="${ALMOND_TTS_CODESIGN_IDENTITY:--}"
PYINSTALLER_BIN="${PYINSTALLER_BIN:-}"

usage() {
  cat <<EOF
Usage: ${0##*/} [--no-codesign] [--no-dmg]

Environment overrides:
  ALMOND_TTS_ICON_PATH          Path to .icns icon (optional)
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

echo "==> Building via PyInstaller"
pushd "$PROJECT_ROOT" >/dev/null
"$PYINSTALLER_BIN" "$SPEC_FILE"
popd >/dev/null

CLI_BIN=""
if [[ -x "$CLI_TARGET/almond_tts" ]]; then
  CLI_BIN="$CLI_TARGET/almond_tts"
elif [[ -x "$CLI_TARGET" ]]; then
  CLI_BIN="$CLI_TARGET"
else
  echo "ERROR: CLI binary not found (looked for $CLI_TARGET/almond_tts and $CLI_TARGET)"
  exit 1
fi

if [[ ! -d "$APP_PATH" ]]; then
  echo "ERROR: App bundle not found at $APP_PATH"
  exit 1
fi

if [[ -n "$ICON_PATH" && -f "$ICON_PATH" ]]; then
  echo "==> Copying custom icon"
  cp "$ICON_PATH" "$APP_PATH/Contents/Resources/AppIcon.icns"
fi

if [[ $NO_CODESIGN -eq 0 ]]; then
  echo "==> Codesigning app bundle with identity: $SIGN_IDENTITY"
  codesign --deep --force --sign "$SIGN_IDENTITY" "$APP_PATH"
else
  echo "==> Skipping codesign (--no-codesign)"
fi

if [[ $NO_DMG -eq 0 ]]; then
  echo "==> Creating DMG"
  rm -f "$DMG_PATH"
  hdiutil create -fs HFS+ -volname AlmondTTS \
    -srcfolder "$APP_PATH" "$DMG_PATH"
else
  echo "==> Skipping DMG creation (--no-dmg)"
fi

echo "==> Build complete"
echo "CLI binary: $CLI_BIN"
echo "App bundle: $APP_PATH"
if [[ $NO_DMG -eq 0 ]]; then
  echo "DMG: $DMG_PATH"
fi
