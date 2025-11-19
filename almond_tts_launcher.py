#!/usr/bin/env python3
"""Launcher used by bundled builds to configure paths before running the CLI."""

from __future__ import annotations

import os
import subprocess
import sys
import shutil
from pathlib import Path

USER_ROOT_ENV = "ALMOND_TTS_USER_ROOT"
USER_DIR_NAMES = ("input", "output", "reference_audio", "working")
FIRST_RUN_SENTINEL = ".almondtts_init"
SAMPLE_NAME = "0000-BasicGrammar.txt"
SAMPLE_RELATIVE_DIRS = ("Samples", "input")
HELP_FLAGS = ("-h", "--help")
HELP_TEXT = """usage: {exe} [-h] [-o OUTPUT_NAME] [-d OUTPUT_DIR]
               [-r REFERENCE_AUDIO] [-l LANGUAGE]
               [--device {{cpu,mps,cuda,auto}}]
               [--min-duration MIN_DURATION]
               [--max-duration MAX_DURATION] [--keep-temp]
               [--workers WORKERS] [--pause-after PAUSE_AFTER]
               [--voice-map VOICE_MAP] [--auto-detect-language]
               input_file

Generate audio from long-form text with intelligent segmentation

positional arguments:
  input_file            Path to input text file or directory containing text files

options:
  -h, --help            show this help message and exit
  -o OUTPUT_NAME, --output-name OUTPUT_NAME
                        Base name for output files (default: input filename without extension)
  -d OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory for output files (default: ~/Documents/AlmondTTS/output)
  -r REFERENCE_AUDIO, --reference-audio REFERENCE_AUDIO
                        Path to reference audio file for voice cloning (optional, uses built-in voice if not provided)
  -l LANGUAGE, --language LANGUAGE
                        Language code (default: es)
  --device {{cpu,mps,cuda,auto}}
                        Device to use for inference (default: auto)
  --min-duration MIN_DURATION
                        Minimum target segment duration in seconds (default: 30)
  --max-duration MAX_DURATION
                        Maximum target segment duration in seconds (default: 60)
  --keep-temp           Keep temporary audio files
  --workers WORKERS     Number of parallel workers for TTS generation (default: 2)
  --pause-after PAUSE_AFTER
                        Add a pause of this many seconds after each audio segment (overrides break tags)
  --voice-map VOICE_MAP
                        JSON mapping of language codes to voice files, e.g., '{{"en": "english.wav", "es": null}}'
  --auto-detect-language
                        Automatically detect language per segment and use the voice from --voice-map

Example usage:
  {exe} input.txt
  {exe} input.txt --min-duration 20 --max-duration 45
  {exe} input.txt --device cpu --keep-temp
"""


def _print_launch_banner() -> None:
    print("Launching AlmondTTS... loading models can take a minute or more on first run. Please wait.")


def _disable_typeguard_instrumentation() -> None:
    try:
        import typeguard  # type: ignore
    except Exception:
        return

    def _noop(target=None, **kwargs):
        if target is None:
            def decorator(func):
                return func
            return decorator
        return target

    try:
        import typeguard._decorators as decorators  # type: ignore
    except Exception:
        decorators = None

    try:
        typeguard.typechecked = _noop  # type: ignore[attr-defined]
    except Exception:
        pass

    if decorators is not None:
        decorators.typechecked = _noop  # type: ignore[attr-defined]


def _set_bundle_environment() -> None:
    """Point the app at bundled resources (if any) without overriding user choices."""
    exe_path = Path(sys.argv[0]).resolve()

    # Look for a macOS .app structure (Contents/Resources)
    contents_dir = None
    for parent in exe_path.parents:
        if parent.name == "Contents":
            contents_dir = parent
            break

    if contents_dir:
        resources_dir = contents_dir / "Resources"
        if resources_dir.exists():
            os.environ.setdefault("ALMOND_TTS_RESOURCES_DIR", str(resources_dir))
            models_dir = resources_dir / "models"
            if models_dir.exists():
                os.environ.setdefault("ALMOND_TTS_MODEL_DIR", str(models_dir))

    # Fallback for CLI/onedir builds where Resources/ doesn't exist
    base_dir = Path(getattr(sys, "_MEIPASS", exe_path.parent))
    models_dir = base_dir / "models"
    if models_dir.exists():
        os.environ.setdefault("ALMOND_TTS_MODEL_DIR", str(models_dir))


def _resolve_user_root() -> Path:
    custom = os.environ.get(USER_ROOT_ENV)
    if custom:
        root = Path(custom).expanduser()
    else:
        root = Path.home() / "Documents" / "AlmondTTS"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prepare_user_directories() -> Path:
    root = _resolve_user_root()
    for sub in USER_DIR_NAMES:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _show_first_run_message(input_dir: Path) -> None:
    message = (
        "AlmondTTS has created the working folders in:\n"
        f"{input_dir.parent}\n\n"
        "A sample file will be placed in the 'input' folder automatically.\n"
        "Add your .txt files to the 'input' folder and run the CLI from /Applications/AlmondTTS.\n"
        "  cd /Applications/AlmondTTS\n"
        f"  ./{Path(sys.argv[0]).name} ~/Documents/AlmondTTS/input/{SAMPLE_NAME}\n"
        "Tip: pass the input folder itself to process every .txt file inside it:\n"
        f"  ./{Path(sys.argv[0]).name} ~/Documents/AlmondTTS/input --device mps\n"
        "\n"
        "You only see this message once."
    )
    print(message)

    try:
        subprocess.run(["open", str(input_dir)], check=False)
    except Exception:
        pass


def _copy_sample_to_input(input_dir: Path) -> None:
    targets = [
        Path(sys.argv[0]).resolve().parent,
        Path(__file__).resolve().parent,
    ]
    for base in targets:
        for rel in SAMPLE_RELATIVE_DIRS:
            candidate = base / rel / SAMPLE_NAME
            if candidate.exists():
                destination = input_dir / SAMPLE_NAME
                if not destination.exists():
                    try:
                        shutil.copy(candidate, destination)
                        print(f"Copied sample file to {destination}")
                    except Exception as exc:
                        print(f"Warning: unable to copy sample file ({exc})")
                return


def _maybe_show_first_run_notice(user_root: Path) -> Path:
    input_dir = user_root / "input"
    sentinel = user_root / FIRST_RUN_SENTINEL
    if not sentinel.exists():
        _show_first_run_message(input_dir)
        try:
            sentinel.touch()
        except Exception:
            pass
        _copy_sample_to_input(input_dir)
    else:
        print(f"Using existing AlmondTTS folders in {user_root}")
        print(f"Add files under {input_dir} or pass a path directly when running the CLI.")
    return input_dir


def _handle_no_arguments() -> None:
    user_root = _prepare_user_directories()
    _maybe_show_first_run_notice(user_root)

    exe_name = Path(sys.argv[0]).name or "almond_tts"
    sample_path = Path.home() / "Documents" / "AlmondTTS" / "input" / "0000-BasicGrammar.txt"
    usage = (
        "No input file provided.\n"
        f"Usage: {exe_name} /path/to/input.txt [options]\n"
        f"       {exe_name} /path/to/folder --device mps\n"
        "Example (after copying the bundled sample):\n"
        f"  {exe_name} {sample_path} --device mps\n"
        "Sample location: /Applications/AlmondTTS/Samples/0000-BasicGrammar.txt\n"
        "Run with --help to see all available options."
    )
    print(usage)
    sys.exit(1)


def _handle_help_request() -> None:
    user_root = _prepare_user_directories()
    _maybe_show_first_run_notice(user_root)
    exe_name = Path(sys.argv[0]).name or "almond_tts"
    print(HELP_TEXT.format(exe=exe_name))
    print("\nNeed more guidance? Open START_HERE.txt inside the AlmondTTS folder.")
    sys.exit(0)


def main() -> None:
    _set_bundle_environment()
    if any(arg in HELP_FLAGS for arg in sys.argv[1:]):
        _handle_help_request()
        return
    if len(sys.argv) <= 1:
        _handle_no_arguments()
        return
    _print_launch_banner()
    _disable_typeguard_instrumentation()

    from almond_tts import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
