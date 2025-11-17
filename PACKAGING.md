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

## 2. Stage Bundled Resources (optional)
`build_app.sh` runs this step for you. Run it manually only if you want to pre-download/refresh the XTTS cache ahead of time so later builds can skip the long download.
```bash
./packaging/prepare_bundle_resources.sh
```

## 3. Build CLI Bundle + DMG (Automated)
```bash
./packaging/build_app.sh
```
This script stages bundled resources, runs PyInstaller, codesigns the CLI binary (ad-hoc by default), and produces a DMG that contains the `AlmondTTS` folder.

Artifacts land in `dist/`:
- CLI directory (script prints exact path; typically `dist/almond_tts/almond_tts`)
- `dist/AlmondTTS` (folder copied from the CLI directory for DMG contents)
- `dist/AlmondTTS.dmg` (if DMG creation not disabled)

### Optional flags
- `./packaging/build_app.sh --no-codesign`
- `./packaging/build_app.sh --no-dmg`

### Manual alternative
If you prefer running each step manually, run:
1. `./packaging/prepare_bundle_resources.sh`
2. `pyinstaller almond_tts.spec`
3. `codesign --force --sign <identity> dist/almond_tts/almond_tts`
4. Copy `dist/almond_tts` to `dist/AlmondTTS` (optional, but yields a nicer folder name in Finder)
5. `hdiutil create -fs HFS+ -volname AlmondTTS -srcfolder dist/AlmondTTS dist/AlmondTTS.dmg`

## 4. Test the Package
Before shipping, validate the artifacts on the same machine:

1. **CLI sanity check (no venv active, direct file path):**
   ```bash
   ./dist/almond_tts path/to/sample.txt --device cpu --output-dir /tmp/almond_test
   ```
   (If PyInstaller produced a directory instead, run the `almond_tts` binary inside it.)
2. **Argument handling check:** run `./dist/almond_tts/almond_tts --help` (or invoke it with no args) and confirm it prints usage guidance, creates the user folders, and exits cleanly.
3. **Fresh user smoke test:** temporarily rename `~/Documents/AlmondTTS` (or log into a Guest account) and re-run the CLI to ensure it recreates the working dirs.
4. **Gatekeeper check (after codesign):**
   ```bash
   spctl --assess --type execute --verbose ./dist/almond_tts/almond_tts
   ```

## 5. Environment Overrides
These environment variables affect both development runs and packaged builds:

- `ALMOND_TTS_USER_ROOT`: root folder for user-facing directories (defaults to `~/Documents/AlmondTTS`)
- `ALMOND_TTS_MODEL_DIR`: explicit XTTS cache location (overrides bundled models)
- `ALMOND_TTS_RESOURCES_DIR`: custom `Resources/` path for nonstandard bundle layouts
- `ALMOND_TTS_CODESIGN_IDENTITY`: identity passed to `codesign` (defaults to ad-hoc "-")

## 6. Runtime Notes
- User files live in `~/Documents/AlmondTTS` even for bundled builds.
- Bundled models sit inside `AlmondTTS/Resources/models` in the packaged CLI folder.

## 7. Publishing Releases on GitHub

### Manual Release Flow
1. Ensure `main` is up to date and tests/smoke checks pass.
2. Run `./packaging/build_app.sh` on macOS so `dist/AlmondTTS.dmg` and the CLI payload are refreshed.
3. Validate the DMG locally (Section 4), then commit/push any changes.
4. Tag the release:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
5. In GitHub, create a release for that tag (or let the tag creation auto-open the draft), upload `dist/AlmondTTS.dmg` (and any other artifacts), add release notes, and publish.

### GitHub Actions (Optional)
If you’d like GitHub to build and attach the DMG automatically:

1. Create `.github/workflows/release.yml` similar to:
   ```yaml
   name: Build AlmondTTS DMG
   on:
     workflow_dispatch:
     push:
       tags:
         - v*
   jobs:
     build:
       runs-on: macos-14
       steps:
         - uses: actions/checkout@v4
         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: '3.12'
         - name: Install deps
           run: |
             python -m venv venv
             source venv/bin/activate
             pip install -r requirements.txt pyinstaller
         - name: Build DMG
           run: |
             source venv/bin/activate
             ./packaging/build_app.sh --no-codesign
         - name: Upload artifact
           uses: actions/upload-artifact@v4
           with:
             name: AlmondTTS.dmg
             path: dist/AlmondTTS.dmg
   ```
2. For a fully signed DMG, inject your signing identity/keychain via GitHub secrets and drop `--no-codesign`.
3. Use `actions/upload-release-asset` (or GitHub’s Release workflow) to attach `dist/AlmondTTS.dmg` to the tag’s release automatically.

This setup keeps local builds simple while enabling repeatable GitHub releases.
