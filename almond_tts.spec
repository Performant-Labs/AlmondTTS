# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None

PROJECT_ROOT = Path(globals().get("__file__", Path.cwd())).resolve().parent
BUNDLE_RESOURCES = PROJECT_ROOT / "bundle_resources"
MODELS_DIR = BUNDLE_RESOURCES / "models"

datas, binaries, hiddenimports = [], [], []


def _include_package(name: str) -> None:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(name)
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)


for package in (
    "TTS",
    "torch",
    "scipy",
    "langdetect",
    "psutil",
    "numpy",
    "gruut",
    "gruut_lang_en",
    "gruut_lang_es",
):
    _include_package(package)

if MODELS_DIR.exists():
    datas.append((str(MODELS_DIR), "Resources/models"))

a = Analysis(
    ["almond_tts_launcher.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

cli_exe = EXE(
    pyz,
    a.scripts,
    [],
    [],
    [],
    exclude_binaries=True,
    name="almond_tts",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    argv_emulation=False,
)

coll = COLLECT(
    cli_exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="almond_tts",
)
