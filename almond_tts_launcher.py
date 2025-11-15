#!/usr/bin/env python3
"""Launcher used by bundled builds to configure paths before running the CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

USER_ROOT_ENV = "ALMOND_TTS_USER_ROOT"
USER_DIR_NAMES = ("input", "output", "reference_audio", "working")
FIRST_RUN_SENTINEL = ".almondtts_init"
LAST_DIR_FILE = ".last_input_dir"


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


def _escape_for_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _show_applescript_alert(message: str, timeout: Optional[int] = None) -> bool:
    script = (
        'display alert "AlmondTTS" message "'
        + _escape_for_applescript(message)
        + '" buttons {"OK"} default button "OK"'
    )
    if timeout:
        script += f" giving up after {timeout}"
    try:
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True, text=True)
        return True
    except Exception:
        return False


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
        "Add your .txt files to the 'input' folder. After you close this message you can choose a file to render."
    )
    shown = False
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("AlmondTTS", message)
        root.destroy()
        shown = True
    except Exception:
        pass

    if not shown:
        shown = _show_applescript_alert(message)

    if not shown:
        print(message)

    try:
        subprocess.run(["open", str(input_dir)], check=False)
    except Exception:
        pass


def _get_initial_dialog_dir(user_root: Path) -> Path:
    cfg = user_root / LAST_DIR_FILE
    if cfg.exists():
        content = cfg.read_text().strip()
        if content:
            candidate = Path(content).expanduser()
            if candidate.exists():
                return candidate
    default_dir = user_root / "input"
    if default_dir.exists():
        return default_dir
    return Path.home()


def _persist_last_dir(path: Path, user_root: Path) -> None:
    try:
        (user_root / LAST_DIR_FILE).write_text(str(path))
    except Exception:
        pass


def _show_initializing_notice() -> None:
    message = "Initializing AlmondTTS... (this may take up to 15 seconds on first launch)"
    script = (
        'display alert "AlmondTTS" message "'
        + _escape_for_applescript(message)
        + '" buttons {"Cancel"} default button "Cancel"'
    )
    try:
        result = subprocess.Popen(["osascript", "-e", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result
    except Exception:
        print(message)
        return None


def _select_input_file(initial_dir: Path) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            filename = filedialog.askopenfilename(
                title="Select a text file",
                initialdir=str(initial_dir),
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
        finally:
            root.destroy()

        if filename:
            return Path(filename)
    except Exception:
        pass

    script = f'''
set defaultFolder to POSIX file "{_escape_for_applescript(str(initial_dir))}"
try
    set chosenFile to choose file with prompt "Select a text file to render:" default location defaultFolder
    POSIX path of chosenFile
on error number -128
    return ""
end try
'''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            path_str = result.stdout.strip()
            if path_str:
                return Path(path_str)
    except Exception:
        pass

    print("Unable to open file chooser UI. Please rerun from Terminal with the input file path.")
    return None


def _handle_no_arguments() -> None:
    _show_initializing_notice()
    user_root = _prepare_user_directories()
    input_dir = user_root / "input"
    sentinel = user_root / FIRST_RUN_SENTINEL
    if not sentinel.exists():
        _show_first_run_message(input_dir)
        try:
            sentinel.touch()
        except Exception:
            pass

    initial_dir = _get_initial_dialog_dir(user_root)
    selected = _select_input_file(initial_dir)
    if not selected:
        return
    _persist_last_dir(selected.parent, user_root)

    sys.argv = [sys.argv[0], str(selected)]
    from tts_multilingual import main as cli_main
    cli_main()


def main() -> None:
    _set_bundle_environment()
    if len(sys.argv) <= 1:
        _handle_no_arguments()
        return

    from tts_multilingual import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
