# AlmondTTS (Developer Guide)

This README is for developers/contributors working on AlmondTTS. End users should follow `START_HERE.md` for app/CLI usage instructions.

## Overview

AlmondTTS generates multilingual audio from long text files using the Coqui XTTS v2 model with voice cloning and multi-voice support. The repository includes the source script (`tts_multilingual.py`), packaging tools, and documentation for building a macOS `.app` bundle and CLI binary.

## Developer Setup

### Prerequisites (macOS)

```bash
brew install python@3.12 portaudio ffmpeg
```

### Create & Activate Virtualenv

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The XTTS v2 model (~2GB) downloads on first run.

## Running from Source

With the virtualenv active:

```bash
python tts_multilingual.py /path/to/input.txt --device auto
```

User-facing directories (`~/Documents/AlmondTTS/input`, `output`, etc.) are created automatically. CLI flags include:
- `--reference-audio`, `--voice-map`, `--auto-detect-language`
- `--min-duration`, `--max-duration`, `--pause-after`
- `--device cpu|mps|cuda|auto`

See `START_HERE.md` for end-user examples.

## Packaging Workflow

Developer packaging instructions (resource prep, PyInstaller build, codesign, DMG creation, testing) are documented in `PACKAGING.md`. Use:

```bash
./packaging/build_app.sh
```

to generate both the standalone CLI binary and `AlmondTTS.app`.

## Key Files
- `tts_multilingual.py`: Main script
- `almond_tts_launcher.py`: Entry point for bundled builds
- `almond_tts.spec`: PyInstaller spec
- `packaging/*.sh`: resource staging + build automation
- `bundle_resources/`: staged assets included in the bundle

## Testing & Troubleshooting

Activate the venv, run the script with sample inputs, and monitor `~/Documents/AlmondTTS/output`. If PyTorch falls back to CPU on Apple Silicon, upgrade Torch packages inside the venv:

```bash
pip install --upgrade torch torchvision torchaudio
```

For packaging issues, consult `PACKAGING.md` and the PyInstaller log (`build/almond_tts/warn-*.txt`). Bugs or regressions should be accompanied by repro steps or tests where possible.

## Support

- End user instructions: `START_HERE.md`
- Packaging and release process: `PACKAGING.md`
- Issues/PRs: use the repository’s standard contribution flow.
