# AlmondTTS – Start Here

Welcome! This guide is for end users who just want to convert text files into multilingual audio using the AlmondTTS app or bundled CLI. You do **not** need to install Python or Homebrew—everything you need is inside the download.

## 1. Install

1. Download `AlmondTTS.dmg` (or `.zip`).
2. Open it and drag `AlmondTTS.app` to your `Applications` folder (or run the CLI binary directly from `dist/almond_tts`).
3. On first launch, macOS may warn you that the app is from an unidentified developer. Click “Open” if prompted.

## 2. Prepare Input Text

1. Launch `AlmondTTS.app` once to let it create the `~/Documents/AlmondTTS/*` folders (you’ll see a confirmation dialog); the CLI binary does the same on first run.
2. Open Finder and go to `~/Documents/AlmondTTS/input`.
2. Drop your `.txt` file(s) there. Example format:
   ```
   <voice lang="en">
   Hello! This sentence uses the default English voice.
   </voice>
   <break time="2s">
   <voice lang="es">
   ¡Hola! Esta parte usa la voz en español.
   </voice>
   ```
   - `<voice lang="xx">` switches languages/voices.
   - `<break time="1.5s">` inserts pauses.

## 3. Run the App

1. Double-click `AlmondTTS.app` (you'll briefly see an "Initializing..." notice while it brings up the picker).
2. When the file chooser appears, select the `.txt` file you want to render. AlmondTTS remembers the last folder you used (falls back to your home folder if it no longer exists).
3. The app processes each segment and shows progress in the console window.

## 4. Find Your Audio

Outputs appear in `~/Documents/AlmondTTS/output`:
- Individual segment WAVs: `filename_000.wav`, `filename_001.wav`, …
- Final merged file: `filename.wav`

Reference audio samples for voice cloning can be stored in `~/Documents/AlmondTTS/reference_audio`.

## 5. Optional Settings

From the CLI (inside the bundle or if you prefer Terminal):

```bash
/path/to/AlmondTTS.app/Contents/MacOS/almond_tts \
  input.txt \
  --reference-audio my_voice.wav \
  --device mps \
  --voice-map '{"en": "my_voice.wav", "es": null}'
```

Key options:
- `--reference-audio`: clone a voice for all languages.
- `--voice-map`: JSON mapping of language codes to specific voice files.
- `--auto-detect-language`: automatically pick voices based on detected language.
- `--device`: `mps` (Apple GPU), `cpu`, `cuda` (if available), or `auto`.

## 6. Troubleshooting

- **No output files?** Ensure your `.txt` file is in `~/Documents/AlmondTTS/input` and that it contains text (not just tags).
- **App can’t open?** Right-click the app, choose “Open” once to bypass Gatekeeper.
- **Using CPU instead of GPU?** Apple Silicon automatically uses `mps`. If forced to CPU, rerun with `--device mps`.
- **Need more voices?** Add additional `<voice>` blocks or provide custom reference WAVs.

## 7. Support

If you encounter issues:
- Check the console logs shown when the app runs.
- Ensure you have at least 8 GB of free memory and disk space (models are ~2 GB).
- Contact the developer/maintainer with the log output and any sample input files.

Happy audiobook making!
