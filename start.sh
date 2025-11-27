#!/usr/bin/env bash
# Simple interactive wizard for AlmondTTS CLI.
# Prompts for common arguments and falls back to sensible defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

prompt() {
  local message="$1" default="$2" var
  if [[ -n "$default" ]]; then
    read -r -p "$message [$default]: " var || exit 1
    echo "${var:-$default}"
  else
    read -r -p "$message: " var || exit 1
    echo "$var"
  fi
}

main() {
  local default_input="${HOME}/Documents/AlmondTTS/input/0000-BasicGrammar.txt"
  [[ -f "$default_input" ]] || default_input="${HOME}/Documents/AlmondTTS/input"

  local input_path output_dir language device workers pause_after auto_detect voice_map ref_audio mx_voice_map
  local confirm

  input_path=$(prompt "Input file or folder" "$default_input")
  output_dir=$(prompt "Output directory" "${HOME}/Documents/AlmondTTS/output")
  language=$(prompt "Default language code" "es")
  device=$(prompt "Device (cpu|mps|cuda|auto)" "auto")
  workers=$(prompt "Workers" "1")
  pause_after=$(prompt "Default pause (seconds, blank = only tags)" "")
  auto_detect=$(prompt "Auto-detect language per segment? (y/N)" "n")

  ref_audio=""
  mx_voice_map=""
  if [[ -f "${SCRIPT_DIR}/reference_audio/EmotionalIntelligenceClip.wav" ]]; then
    mx_voice_map="{\"es\": \"${SCRIPT_DIR}/reference_audio/EmotionalIntelligenceClip.wav\"}"
    ref_audio="${SCRIPT_DIR}/reference_audio/EmotionalIntelligenceClip.wav"
  fi
  voice_map=$(prompt "Voice map JSON" "$mx_voice_map")

  echo
  echo "Launching AlmondTTS with:"
  echo "  input       : $input_path"
  echo "  output dir  : $output_dir"
  echo "  language    : $language"
  echo "  device      : $device"
  echo "  workers     : $workers"
  [[ -n "$pause_after" ]] && echo "  pause after : $pause_after s" || echo "  pause after : (use tags/defaults)"
  [[ -n "$voice_map" ]] && echo "  voice map   : $voice_map" || echo "  voice map   : (none)"
  [[ "$auto_detect" =~ ^[Yy]$ ]] && echo "  auto-detect : enabled" || echo "  auto-detect : disabled"
  echo

  cmd=(python3 "${SCRIPT_DIR}/almond_tts.py" "$input_path" --output-dir "$output_dir" --language "$language" --workers "$workers" --device "$device")
  [[ -n "$pause_after" ]] && cmd+=(--pause-after "$pause_after")
  [[ -n "$voice_map" ]] && cmd+=(--voice-map "$voice_map")
  [[ "$auto_detect" =~ ^[Yy]$ ]] && cmd+=(--auto-detect-language)

  echo "Command:"
  printf '  %q' "${cmd[@]}"
  echo

  read -r -p "Proceed with these settings? (y/N): " confirm || exit 1
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi

  "${cmd[@]}"
}

main "$@"
