#!/bin/bash

# Setup script for TTS with Breaks
set -e

echo "=========================================="
echo "Setting up TTS with Breaks"
echo "=========================================="
echo ""

cd ~/Projects/AlmondTTS

echo "[1] Creating virtual environment..."
python3.12 -m venv venv
echo "✓ Virtual environment created"
echo ""

echo "[2] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

echo "[3] Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

echo "[4] Installing Python packages..."
echo "This will take 5-10 minutes (downloading PyTorch ~2GB)..."
echo ""
pip install -r requirements.txt
echo ""
echo "✓ All packages installed"
echo ""

echo "[5] Verifying installation..."
python -c "import numpy; print('✓ numpy:', numpy.__version__)"
python -c "import scipy; print('✓ scipy:', scipy.__version__)"
python -c "import TTS; print('✓ TTS installed')"
python -c "import torch; print('✓ torch:', torch.__version__)"
python - <<'PY'
try:
    from importlib import metadata
    print("✓ langdetect:", metadata.version("langdetect"))
except Exception as exc:
    print("langdetect check failed:", exc)
PY
echo ""

echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To use the script:"
echo "  1. Activate the virtual environment:"
echo "     cd ~/Projects/AlmondTTS"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the script:"
echo "     python tts_with_breaks.py example_input.txt"
echo ""
echo "  3. When done, deactivate:"
echo "     deactivate"
echo ""
echo "Output will be saved to: ~/Documents/AlmondTTS/output/"
echo "=========================================="
