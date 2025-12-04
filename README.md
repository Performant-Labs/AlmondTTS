# AlmondTTS

AlmondTTS turns long-form text into natural-sounding multilingual audio, with
voice cloning and multi-voice support. It is built on the open-source Coqui TTS
engine (`TTS`), and this project depends on the ongoing work of the
community-maintained [Idiap fork](https://github.com/idiap/coqui-ai-TTS) from
[Idiap Research Institute](https://www.idiap.ch/).

- **End users**: start with `START_HERE.txt` in the download folder for a
  step-by-step quick start guide.
- **Developers / contributors**: see `CONTRIBUTING.md` for setup, running from
  source, and packaging details.

---

## What AlmondTTS Can Do

- **Long-form TTS**
  - Convert long text files (articles, chapters, scripts) into continuous audio.
  - Intelligent segmentation targets ~3060 seconds per segment so playback
    sounds natural.

- **Multilingual XTTS v2 engine**
  - Uses the Coqui XTTS v2 model.
  - Supports multiple languages (including English and Spanish) in a single
    project.

- **Voice cloning and multi-voice**
  - Clone a voice from a short reference audio clip.
  - Use different voices or languages for different parts of the same script
    via tags and configuration.

- **Flexible timing**
  - Control minimum/maximum segment duration.
  - Insert pauses between segments or at explicit `<break>` tags.
  - Optionally slow down or speed up the final audio via `--slowdown-factor` without re-running TTS.
  - Choose between `rubberband` (default, higher quality for voice) or `sox` engines via `--slowdown-engine`.

For concrete usage examples and terminal commands, open `START_HERE.txt`.

---

## Where to Go Next

- **I just want to use AlmondTTS**
  - Download the latest **macOS** release from the
    [GitHub Releases page](https://github.com/Performant-Labs/AlmondTTS/releases).
  - Open the downloaded `.dmg` (or `.zip`) file and drag the `AlmondTTS` folder
    into your `Applications` folder (or another folder you prefer).
  - Read and follow `START_HERE.txt` inside the `AlmondTTS` folder to install
    the app bundle, find the input folder, and generate your first audio.
  - Prefer a guided setup? Run `./start.sh` for an interactive wizard.
  - The wizard includes prompts for slowdown factor and engine (rubberband or sox).
  - AlmondTTS is currently available for macOS only.

- **I want to tweak or extend AlmondTTS**
  - Read `CONTRIBUTING.md` for:
    - Local development setup (Python/venv/requirements).
    - Running `almond_tts.py` from source.
    - Packaging with PyInstaller and DMG creation.

---

## Support

- End user quick start and examples: `START_HERE.txt`.
- Developer setup, contributing, and packaging: `CONTRIBUTING.md` and
  `PACKAGING.md`.
- Issues/PRs: use the repository's standard contribution flow.
