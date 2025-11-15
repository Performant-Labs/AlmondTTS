# AlmondTTS Packaging Guide

## 1. Prerequisites
- macOS (arm64 build recommended)
- Xcode command-line tools (`xcode-select --install`)
- Activated project virtualenv with runtime deps installed
- PyInstaller installed inside that environment:
  ```bash
  source venv/bin/activate
  pip install pyinstaller
  ```

## 2. Stage Bundled Resources
Preload the XTTS cache so the `.app` is self-contained:
```bash
./packaging/prepare_bundle_resources.sh
```

## 3. Build CLI + App Bundle (Automated)
```bash
./packaging/build_app.sh
```
This script stages bundled resources, runs PyInstaller, codesigns the app (ad-hoc by default), and produces a DMG.

Artifacts land in `dist/`:
- CLI binary (script prints exact path; typically `dist/almond_tts`)
- `dist/AlmondTTS.app` (double-clickable app)
- `dist/AlmondTTS.dmg` (if DMG creation not disabled)

### Optional flags
- `./packaging/build_app.sh --no-codesign`
- `./packaging/build_app.sh --no-dmg`

### Manual alternative
If you prefer running each step manually, run:
1. `./packaging/prepare_bundle_resources.sh`
2. `pyinstaller almond_tts.spec`
3. `codesign --deep --force --sign <identity> dist/AlmondTTS.app`
4. `hdiutil create -fs HFS+ -volname AlmondTTS -srcfolder dist/AlmondTTS.app dist/AlmondTTS.dmg`

## 4. Test the Package
Before shipping, validate the artifacts on the same machine:

1. **CLI sanity check (no venv active):**
   ```bash
   ./dist/almond_tts path/to/sample.txt --device cpu --output-dir /tmp/almond_test
   ```
   (If PyInstaller produced a directory instead, run the `almond_tts` binary inside it.)
2. **App bundle launch:**
   ```bash
   open dist/AlmondTTS.app
   ```
   Drop a text file into `~/Documents/AlmondTTS/input` and confirm output appears in `~/Documents/AlmondTTS/output`.
3. **Fresh user smoke test:** temporarily rename `~/Documents/AlmondTTS` (or log into a Guest account) and relaunch the app to ensure it recreates the working dirs.
4. **Gatekeeper check (after codesign):**
   ```bash
   spctl --assess --type execute --verbose dist/AlmondTTS.app
   ```

## 5. Environment Overrides
These environment variables affect both development runs and packaged builds:

- `ALMOND_TTS_USER_ROOT`: root folder for user-facing directories (defaults to `~/Documents/AlmondTTS`)
- `ALMOND_TTS_MODEL_DIR`: explicit XTTS cache location (overrides bundled models)
- `ALMOND_TTS_RESOURCES_DIR`: custom `Resources/` path for nonstandard bundle layouts
- `ALMOND_TTS_ICON_PATH`: `.icns` path consumed by `build_app.sh`
- `ALMOND_TTS_CODESIGN_IDENTITY`: identity passed to `codesign` (defaults to ad-hoc "-")

## 6. Runtime Notes
- User files live in `~/Documents/AlmondTTS` even for bundled builds.
- Bundled models sit inside `AlmondTTS.app/Contents/Resources/models`.
