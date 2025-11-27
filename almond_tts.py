#!/usr/bin/env python3
"""
Long-Form TTS - Generate audio from long text with intelligent segmentation

This script uses a two-phase approach:
1. Phase 1: Analyze and segment text into optimal chunks (30-60s target)
2. Phase 2: Generate audio for each segment and concatenate
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import warnings
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psutil
from scipy.io import wavfile
from TTS.api import TTS

# Suppress torchaudio deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')

APP_NAME = "AlmondTTS"
USER_ROOT_ENV = "ALMOND_TTS_USER_ROOT"
MODEL_DIR_ENV = "ALMOND_TTS_MODEL_DIR"
RESOURCES_DIR_ENV = "ALMOND_TTS_RESOURCES_DIR"
USER_FACING_DIRS = ("input", "output", "reference_audio", "working")


def _expand_path(path_str: str) -> Path:
    """Expand user/home references in a path string."""
    return Path(path_str).expanduser()


@lru_cache(maxsize=1)
def get_bundle_resources_dir() -> Optional[Path]:
    """Return the Resources directory when running inside a macOS .app bundle."""
    env_path = os.environ.get(RESOURCES_DIR_ENV)
    if env_path:
        candidate = _expand_path(env_path)
        if candidate.exists():
            return candidate

    if hasattr(sys, "_MEIPASS"):
        candidate = Path(sys._MEIPASS)
        resources = candidate / "Resources"
        if resources.exists():
            return resources
        if candidate.exists():
            return candidate

    exe_path = Path(sys.argv[0]).resolve()
    for parent in exe_path.parents:
        if parent.name == "Contents":
            resources = parent / "Resources"
            if resources.exists():
                return resources
    return None


@lru_cache(maxsize=1)
def get_user_root_dir() -> Path:
    """Directory exposed to users for inputs/outputs (defaults to ~/Documents/AlmondTTS)."""
    env_path = os.environ.get(USER_ROOT_ENV)
    if env_path:
        root = _expand_path(env_path)
    else:
        root = Path.home() / "Documents" / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_user_subdir(name: str) -> Path:
    """Create and return a subdirectory under the user root."""
    subdir = get_user_root_dir() / name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def ensure_user_directories() -> None:
    """Guarantee that user-facing directories exist before processing."""
    for directory in USER_FACING_DIRS:
        get_user_subdir(directory)


@lru_cache(maxsize=1)
def get_model_cache_dir() -> Path:
    """Determine where the XTTS model cache should live."""
    env_path = os.environ.get(MODEL_DIR_ENV)
    if env_path:
        model_dir = _expand_path(env_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    bundle_resources = get_bundle_resources_dir()
    if bundle_resources:
        bundle_models = bundle_resources / "models"
        if bundle_models.exists():
            return bundle_models

    legacy_dir = Path.home() / "Projects" / "tts-models"
    if legacy_dir.exists():
        return legacy_dir

    model_dir = get_user_root_dir() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

@dataclass
class TextSegment:
    """Represents a segment of text to be processed"""
    text: str
    break_after: float  # Seconds of silence after this segment
    estimated_duration: float  # Estimated audio duration in seconds
    segment_id: int
    voice_file: Optional[str] = None  # Path to reference audio for voice cloning (None = use default voice)
    language: Optional[str] = None  # Language code for this segment


class LongFormTTS:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                 speaker_wav=None, language="es", output_dir=None, device=None, num_workers=1, pause_after=2.0,
                 voice_map=None, auto_detect_language=False):
        """
        Initialize the Long-Form TTS processor.

        Args:
            model_name: The Coqui TTS model to use
            speaker_wav: Path to reference audio file for voice cloning (default voice)
            language: Default language code
            output_dir: Directory to save output files
            device: Device to use ('cpu', 'mps', 'cuda', or None for auto-detect)
            num_workers: Number of parallel workers for TTS generation (default: 1)
            pause_after: If set, add this many seconds of pause after each audio segment (default: 2.0). Break tags always take precedence.
            voice_map: Dict mapping language codes to voice files, e.g., {"en": "english.wav", "es": None}
            auto_detect_language: If True, automatically detect language per segment
        """
        init_parts = [f"AlmondTTS start", f"model: {model_name}", f"lang: {language}"]
        if speaker_wav:
            init_parts.append(f"voice: {speaker_wav}")
        print(" | ".join(init_parts))

        self.speaker_wav = speaker_wav
        self.language = language
        self.model_name = model_name
        self.voice_map = voice_map or {}
        self.auto_detect_language = auto_detect_language

        if voice_map:
            print(f"Voice mapping enabled:")
            for lang, voice in voice_map.items():
                voice_str = voice if voice else "default voice"
                print(f"  {lang}: {voice_str}")

        if auto_detect_language:
            print(f"Auto language detection: ENABLED")
            try:
                from langdetect import detect, DetectorFactory
                DetectorFactory.seed = 0  # For consistent results
                self._detect_language = detect
            except ImportError:
                print("WARNING: langdetect not installed. Auto-detection disabled.")
                print("Install with: pip install langdetect")
                self.auto_detect_language = False
                self._detect_language = None
        else:
            self._detect_language = None

        # Set up persistent model cache directory
        cache_dir = get_model_cache_dir()
        os.environ['COQUI_TTS_CACHE_DIR'] = str(cache_dir)
        print(f"Model cache directory: {cache_dir}")

        # Check for GPU availability or use specified device
        import torch
        if device is None:
            # Auto-detect
            if torch.cuda.is_available():
                device = "cuda"
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
                print("Using Apple Silicon GPU (MPS)")
            else:
                device = "cpu"
                print("Using CPU (GPU not available)")
        else:
            # Use specified device
            if device == "mps" and not torch.backends.mps.is_available():
                print(f"Warning: MPS requested but not available, falling back to CPU")
                device = "cpu"
            elif device == "cuda" and not torch.cuda.is_available():
                print(f"Warning: CUDA requested but not available, falling back to CPU")
                device = "cpu"
            print(f"Using device: {device.upper()}")

        # Initialize multiple TTS model instances (one per worker for true parallelism)
        print(f"\n{'='*60}")
        print(f"Memory & Model Loading")
        print(f"{'='*60}")

        # Get initial memory stats
        process = psutil.Process()
        initial_process_mem = process.memory_info().rss / (1024**3)  # GB
        available_mem = psutil.virtual_memory().available / (1024**3)  # GB
        total_mem = psutil.virtual_memory().total / (1024**3)  # GB

        print(f"System RAM: {total_mem:.2f} GB total")
        print(f"Available RAM: {available_mem:.2f} GB")
        print(f"Process memory (before models): {initial_process_mem:.2f} GB")
        print(f"\nLoading {num_workers} TTS model instance(s) for parallel processing...")

        self.tts_models = []
        self.device = device

        for i in range(num_workers):
            mem_before = process.memory_info().rss / (1024**3)
            print(f"\n  [{i+1}/{num_workers}] Loading model...")
            tts_instance = TTS(model_name=model_name).to(device)
            self.tts_models.append(tts_instance)
            mem_after = process.memory_info().rss / (1024**3)
            model_mem = mem_after - mem_before
            print(f"  [{i+1}/{num_workers}] Model loaded: +{model_mem:.2f} GB (process total: {mem_after:.2f} GB)")

        final_process_mem = process.memory_info().rss / (1024**3)
        total_model_mem = final_process_mem - initial_process_mem
        available_after = psutil.virtual_memory().available / (1024**3)

        print(f"\n{'='*60}")
        print(f"All {num_workers} model(s) loaded successfully")
        print(f"Total model memory: {total_model_mem:.2f} GB")
        print(f"Process memory (after models): {final_process_mem:.2f} GB")
        print(f"Available RAM remaining: {available_after:.2f} GB")
        print(f"{'='*60}\n")

        # Create a queue of available model indices
        import queue
        self.model_queue = queue.Queue()
        for i in range(num_workers):
            self.model_queue.put(i)

        # Set output directory
        if output_dir is None:
            self.output_dir = get_user_subdir("output")
        else:
            self.output_dir = _expand_path(str(output_dir))
            self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        self.temp_files = []
        self.silence_cache = {}  # Cache silence files by duration
        self.num_workers = num_workers
        self.pause_after = pause_after
        print(f"Parallel workers: {num_workers}")
        if pause_after is not None:
            print(f"Pause after each segment (unless break tag present): {pause_after}s")

        # Estimation: average speaking rate for Spanish TTS
        # Conservative estimate to avoid segments being too long
        # Adjusted based on actual TTS output (was 2.5, but TTS speaks slower)
        self.words_per_second = 1.5

    def estimate_duration(self, text: str) -> float:
        """
        Estimate audio duration based on word count.

        Args:
            text: Text to estimate duration for

        Returns:
            Estimated duration in seconds
        """
        word_count = len(text.split())
        # Use a more conservative rate when pause_after is set to keep segments shorter
        rate = 1.0 if self.pause_after is not None else self.words_per_second
        return word_count / rate

    def split_by_punctuation(self, text: str, max_duration: float = 60.0) -> List[str]:
        """
        Split text by punctuation to keep segments under max_duration.

        Args:
            text: Text to split
            max_duration: Maximum duration in seconds

        Returns:
            List of text chunks
        """
        # Try splitting by major punctuation first (semicolon, colon, em-dash)
        major_punct = r'([;:\u2014]\s+)'
        parts = re.split(major_punct, text)

        chunks = []
        current_chunk = ""

        for part in parts:
            test_chunk = current_chunk + part
            if self.estimate_duration(test_chunk) > max_duration and current_chunk:
                # Current chunk would be too long, save what we have
                chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk = test_chunk

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If still too long, split by commas
        final_chunks = []
        for chunk in chunks:
            if self.estimate_duration(chunk) > max_duration:
                comma_parts = re.split(r'(,\s+)', chunk)
                sub_chunk = ""
                for part in comma_parts:
                    test_sub = sub_chunk + part
                    if self.estimate_duration(test_sub) > max_duration and sub_chunk:
                        final_chunks.append(sub_chunk.strip())
                        sub_chunk = part
                    else:
                        sub_chunk = test_sub
                if sub_chunk.strip():
                    final_chunks.append(sub_chunk.strip())
            else:
                final_chunks.append(chunk)

        return [c for c in final_chunks if c]

    def segment_text(self, text: str, min_duration: float = 30.0, max_duration: float = 60.0) -> List[TextSegment]:
        """
        Phase 1: Analyze and segment text into optimal chunks.

        Args:
            text: Input text to segment
            min_duration: Minimum target duration in seconds
            max_duration: Maximum target duration in seconds

        Returns:
            List of TextSegment objects
        """
        print("\n=== PHASE 1: Text Analysis and Segmentation ===")

        # First, handle break tags (supports both <break time="5s"> and <break time="5s"/>)
        break_pattern = r'<break\s+time=["\']([\d.]+)s["\']\s*/?>'
        parts = re.split(break_pattern, text)

        segments = []
        segment_id = 0

        for i in range(0, len(parts)):
            if i % 2 == 0:  # This is text content
                text_part = parts[i].strip()
                if not text_part:
                    continue

                # Get break time that follows
                break_time = 0.0
                if i + 1 < len(parts):
                    break_time = float(parts[i + 1])

                # Split into sentences
                sentence_pattern = r'([.!?]+[\s"\']*)'
                sentences = re.split(sentence_pattern, text_part)

                current_text = ""
                for j in range(0, len(sentences), 2):
                    if j >= len(sentences):
                        break

                    sentence = sentences[j]
                    punct = sentences[j + 1] if j + 1 < len(sentences) else ""
                    full_sentence = (sentence + punct).strip()

                    if not full_sentence:
                        continue

                    # Check if adding this sentence keeps us in range
                    test_text = (current_text + " " + full_sentence).strip()
                    estimated = self.estimate_duration(test_text)

                    if estimated > max_duration:
                        # Current text would be too long, save what we have (if any)
                        if current_text:
                            segments.append(TextSegment(
                                text=current_text,
                                break_after=0.0,
                                estimated_duration=self.estimate_duration(current_text),
                                segment_id=segment_id
                            ))
                            segment_id += 1
                        current_text = full_sentence

                        # For pause_after mode, if even a single sentence is too long, split it by words
                        if self.pause_after is not None and self.estimate_duration(current_text) > max_duration:
                            words = current_text.split()
                            temp_chunk = ""
                            for word in words:
                                test_word = (temp_chunk + " " + word).strip()
                                if self.estimate_duration(test_word) > max_duration and temp_chunk:
                                    segments.append(TextSegment(
                                        text=temp_chunk,
                                        break_after=0.0,
                                        estimated_duration=self.estimate_duration(temp_chunk),
                                        segment_id=segment_id
                                    ))
                                    segment_id += 1
                                    temp_chunk = word
                                else:
                                    temp_chunk = test_word
                            current_text = temp_chunk
                    elif estimated < min_duration:
                        # Keep accumulating until we reach min
                        # But for pause_after mode, enforce strict max_duration limit
                        if self.pause_after is not None and estimated >= max_duration:
                            # Adding this sentence would exceed max, save current and start fresh
                            if current_text:
                                segments.append(TextSegment(
                                    text=current_text,
                                    break_after=0.0,
                                    estimated_duration=self.estimate_duration(current_text),
                                    segment_id=segment_id
                                ))
                                segment_id += 1
                            current_text = full_sentence
                        else:
                            current_text = test_text
                    else:
                        # We're in the acceptable range - save this segment and start a new one
                        segments.append(TextSegment(
                            text=test_text,
                            break_after=0.0,
                            estimated_duration=estimated,
                            segment_id=segment_id
                        ))
                        segment_id += 1
                        current_text = ""

                # Save remaining text
                if current_text.strip():
                    # Check if this segment is too long and needs splitting
                    if self.estimate_duration(current_text) > max_duration:
                        # For pause_after mode with short segments, split more aggressively
                        if self.pause_after is not None and max_duration <= 10:
                            # Split by words to create very short segments
                            words = current_text.split()
                            temp_chunk = ""
                            for word in words:
                                test = (temp_chunk + " " + word).strip()
                                if self.estimate_duration(test) > max_duration and temp_chunk:
                                    segments.append(TextSegment(
                                        text=temp_chunk,
                                        break_after=0.0,
                                        estimated_duration=self.estimate_duration(temp_chunk),
                                        segment_id=segment_id
                                    ))
                                    segment_id += 1
                                    temp_chunk = word
                                else:
                                    temp_chunk = test
                            if temp_chunk:
                                segments.append(TextSegment(
                                    text=temp_chunk,
                                    break_after=break_time,
                                    estimated_duration=self.estimate_duration(temp_chunk),
                                    segment_id=segment_id
                                ))
                                segment_id += 1
                        else:
                            # Use normal punctuation-based splitting for longer segments
                            sub_chunks = self.split_by_punctuation(current_text, max_duration)
                            for idx, chunk in enumerate(sub_chunks):
                                # Apply break_time only to the last sub-chunk
                                is_last_chunk = (idx == len(sub_chunks) - 1)
                                segments.append(TextSegment(
                                    text=chunk,
                                    break_after=break_time if is_last_chunk else 0.0,
                                    estimated_duration=self.estimate_duration(chunk),
                                    segment_id=segment_id
                                ))
                                segment_id += 1
                    else:
                        segments.append(TextSegment(
                            text=current_text,
                            break_after=break_time,
                            estimated_duration=self.estimate_duration(current_text),
                            segment_id=segment_id
                        ))
                        segment_id += 1

        # Print segmentation summary
        print(f"\nSegmentation complete:")
        print(f"  Total segments: {len(segments)}")
        total_estimated = sum(s.estimated_duration for s in segments)
        print(f"  Estimated total audio: {total_estimated:.1f}s ({total_estimated/60:.1f} minutes)")
        print(f"  Average segment length: {total_estimated/len(segments):.1f}s")

        # Show segment details
        print(f"\nSegment breakdown:")
        for seg in segments:
            preview = seg.text[:60] + "..." if len(seg.text) > 60 else seg.text
            break_info = f" + {seg.break_after}s break" if seg.break_after > 0 else ""
            print(f"  [{seg.segment_id:03d}] ~{seg.estimated_duration:.1f}s{break_info}: {preview}")

        return segments

    def parse_text_with_breaks(self, filepath: str) -> List[TextSegment]:
        """
        Parse text file with explicit <break time="Xs"> and <voice> tags.
        Returns list of TextSegment objects.

        Supported tags:
        - <break time="2s"> - Add a pause
        - <voice ref="path/to/voice.wav" lang="en">text</voice> - Use specific voice and language
        - <voice lang="es">text</voice> - Use default voice for language

        Args:
            filepath: Path to input text file

        Returns:
            List of TextSegment objects
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # First, extract voice tags and replace with markers
        voice_pattern = r'<voice\s+(?:ref=["\'](.*?)["\']\s+)?(?:lang=["\'](.*?)["\']\s*)?>(.*?)</voice>'
        voice_segments = []

        def voice_replacer(match):
            voice_ref = match.group(1)  # Can be None
            lang = match.group(2)  # Can be None
            text = match.group(3)
            marker = f"__VOICE_SEGMENT_{len(voice_segments)}__"
            voice_segments.append((text, voice_ref, lang))
            return marker

        content = re.sub(voice_pattern, voice_replacer, content, flags=re.DOTALL)

        # Split by break tags (supports both <break time="5s"> and <break time="5s"/>)
        break_pattern = r'<break\s+time=["\']([\d.]+)s["\']\s*/?>'
        parts = re.split(break_pattern, content)

        segments = []
        segment_id = 0

        # Process pairs of (text, break_time)
        for i in range(0, len(parts), 2):
            text = parts[i].strip()
            if not text:
                continue

            # Get break time (0 if last segment or no break specified)
            break_time = float(parts[i + 1]) if i + 1 < len(parts) else 0.0

            # Check if this text contains voice segment markers
            voice_marker_pattern = r'__VOICE_SEGMENT_(\d+)__'
            voice_matches = list(re.finditer(voice_marker_pattern, text))

            if voice_matches:
                # Process each voice segment separately
                last_end = 0
                for match in voice_matches:
                    # Add any text before the voice marker as a regular segment
                    before_text = text[last_end:match.start()].strip()
                    if before_text:
                        segments.append(self._create_segment(
                            before_text, 0.0, segment_id, None, None
                        ))
                        segment_id += 1

                    # Add the voice segment
                    voice_idx = int(match.group(1))
                    voice_text, voice_ref, lang = voice_segments[voice_idx]
                    voice_text = voice_text.strip()

                    if voice_text:
                        # Determine break time (only on the last voice segment in this part)
                        is_last = match == voice_matches[-1]
                        seg_break = break_time if is_last else 0.0

                        segments.append(self._create_segment(
                            voice_text, seg_break, segment_id, voice_ref, lang
                        ))
                        segment_id += 1

                    last_end = match.end()

                # Add any remaining text after the last voice marker
                after_text = text[last_end:].strip()
                if after_text:
                    segments.append(self._create_segment(
                        after_text, break_time, segment_id, None, None
                    ))
                    segment_id += 1
            else:
                # No voice tags, create regular segment
                segments.append(self._create_segment(
                    text, break_time, segment_id, None, None
                ))
                segment_id += 1

        print(f"\nParsed {len(segments)} segments")
        for seg in segments:
            preview = seg.text[:60] + "..." if len(seg.text) > 60 else seg.text
            break_info = f" + {seg.break_after}s break" if seg.break_after > 0 else ""
            voice_info = ""
            if seg.voice_file:
                voice_info = f" [voice: {Path(seg.voice_file).name}]"
            elif seg.language and seg.language != self.language:
                voice_info = f" [lang: {seg.language}]"
            print(f"  [{seg.segment_id:03d}] ~{seg.estimated_duration:.1f}s{break_info}{voice_info}: {preview}")

        return segments

    def _create_segment(self, text: str, break_after: float, segment_id: int,
                       voice_ref: Optional[str], lang: Optional[str]) -> TextSegment:
        """
        Helper to create a TextSegment with voice/language detection.

        Args:
            text: Segment text
            break_after: Pause duration after segment
            segment_id: Segment ID
            voice_ref: Optional voice reference file path
            lang: Optional language code

        Returns:
            TextSegment object
        """
        # Determine language
        if lang:
            # Explicit language from voice tag
            segment_lang = lang
        elif self.auto_detect_language and self._detect_language:
            # Auto-detect language
            try:
                segment_lang = self._detect_language(text)
            except:
                segment_lang = self.language  # Fallback to default
        else:
            # Use default language
            segment_lang = self.language

        # Determine voice file
        if voice_ref:
            # Explicit voice reference from tag
            segment_voice = voice_ref
        elif segment_lang in self.voice_map:
            # Use mapped voice for this language
            segment_voice = self.voice_map[segment_lang]
        else:
            # Use default voice
            segment_voice = self.speaker_wav

        estimated_duration = self.estimate_duration(text)

        return TextSegment(
            text=text,
            break_after=break_after,
            estimated_duration=estimated_duration,
            segment_id=segment_id,
            voice_file=segment_voice,
            language=segment_lang
        )

    def _process_single_segment(self, segment: TextSegment, output_name: str, file_counter: int):
        """
        Process a single text segment using a dedicated model instance.

        Args:
            segment: TextSegment to process
            output_name: Base output name
            file_counter: File counter for naming

        Returns:
            Tuple of (success, tts_filename, silence_filename, generation_time, audio_duration, error_msg)
        """
        tts_filename = self.output_dir / f"{output_name}_{file_counter:03d}.wav"
        silence_filename = None

        # Get a model instance from the queue (blocks if all models are in use)
        model_idx = self.model_queue.get()

        try:
            start_time = time.time()

            # Use the dedicated model instance for this worker
            tts_model = self.tts_models[model_idx]

            # Use segment-specific voice and language if available
            segment_voice = segment.voice_file if segment.voice_file is not None else self.speaker_wav
            segment_lang = segment.language if segment.language else self.language

            if segment_voice:
                tts_model.tts_to_file(
                    text=segment.text,
                    file_path=str(tts_filename),
                    speaker_wav=segment_voice,
                    language=segment_lang
                )
            else:
                # Use default speaker from model (Dionisio Schuyler for XTTS v2)
                tts_model.tts_to_file(
                    text=segment.text,
                    file_path=str(tts_filename),
                    speaker="Dionisio Schuyler",
                    language=segment_lang
                )

            generation_time = time.time() - start_time

            # Get actual audio duration
            with wave.open(str(tts_filename), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                audio_duration = frames / float(rate)

        # Handle silence if needed
            # Respect explicit break tags; otherwise apply pause_after if set
            if segment.break_after > 0:
                pause_duration = segment.break_after
            elif self.pause_after is not None:
                pause_duration = self.pause_after
            else:
                pause_duration = 0

            if pause_duration > 0:
                silence_filename = self.output_dir / f"{output_name}_{file_counter + 1:03d}.wav"
                self.generate_silence(pause_duration, silence_filename)

            return (True, tts_filename, silence_filename, generation_time, audio_duration, None)

        except Exception as e:
            return (False, None, None, 0, 0, str(e))
        finally:
            # Always return the model to the queue
            self.model_queue.put(model_idx)

    def generate_silence(self, duration_seconds, output_path, sample_rate=24000):
        """
        Generate a WAV file containing silence.
        """
        num_samples = int(duration_seconds * sample_rate)
        silence = np.zeros(num_samples, dtype=np.int16)
        wavfile.write(output_path, sample_rate, silence)

    def generate_audio(self, segments: List[TextSegment], output_name: str) -> List[Path]:
        """
        Phase 2: Generate audio for all segments.

        Args:
            segments: List of TextSegment objects to process
            output_name: Base name for output files

        Returns:
            List of generated audio file paths
        """
        print("\n=== PHASE 2: Audio Generation ===")

        # Warmup: Generate a dummy audio to initialize the first model
        print("\nWarming up first model...")
        warmup_start = time.time()
        warmup_file = self.output_dir / "_warmup_temp.wav"
        try:
            warmup_lang = self.language
            warmup_voice = None
            if warmup_lang in self.voice_map:
                warmup_voice = self.voice_map[warmup_lang]
            elif self.speaker_wav:
                warmup_voice = self.speaker_wav

            if warmup_voice:
                self.tts_models[0].tts_to_file(
                    text="Warmup test.",
                    file_path=str(warmup_file),
                    speaker_wav=warmup_voice,
                    language=warmup_lang
                )
            else:
                self.tts_models[0].tts_to_file(
                    text="Warmup test.",
                    file_path=str(warmup_file),
                    speaker="Dionisio Schuyler",
                    language=warmup_lang
                )
            warmup_time = time.time() - warmup_start
            print(f"Model warmup complete ({warmup_time:.2f}s)")
            # Delete warmup file
            if warmup_file.exists():
                warmup_file.unlink()
        except KeyboardInterrupt:
            print("\nWarmup interrupted by user.")
            if warmup_file.exists():
                try:
                    warmup_file.unlink()
                except Exception:
                    pass
            raise
        except Exception as e:
            print(f"Warning: Warmup failed: {e}")

        self.temp_files = []

        # Track timing statistics
        generation_times = []
        audio_durations = []

        # Create list to store results in order
        results = [None] * len(segments)

        # Precompute task metadata to preserve output ordering and filenames
        tasks = []
        file_counter = 0
        for idx, segment in enumerate(segments):
            if segment.break_after > 0:
                pause_duration = segment.break_after
            elif self.pause_after is not None:
                pause_duration = self.pause_after
            else:
                pause_duration = 0
            will_add_pause = pause_duration > 0
            tasks.append({
                "idx": idx,
                "segment": segment,
                "file_counter": file_counter,
                "lang": segment.language if segment.language else self.language
            })
            file_counter += 2 if will_add_pause else 1

        # Process segments language-by-language to reduce cross-language hallucination
        language_order = []
        for task in tasks:
            lang = task["lang"]
            if lang not in language_order:
                language_order.append(lang)

        print(
            f"\nProcessing {len(segments)} segments across {len(language_order)} language group(s) "
            f"with {self.num_workers} parallel worker(s)...\n"
        )

        completed = 0
        for lang in language_order:
            lang_tasks = [t for t in tasks if t["lang"] == lang]
            print(f"  Language '{lang}': {len(lang_tasks)} segment(s)")

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_task = {}
                for task in lang_tasks:
                    future = executor.submit(
                        self._process_single_segment,
                        task["segment"],
                        output_name,
                        task["file_counter"]
                    )
                    future_to_task[future] = task

                try:
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        idx = task["idx"]
                        segment = task["segment"]
                        completed += 1
                        success, tts_file, silence_file, gen_time, audio_dur, error = future.result()

                        if success:
                            results[idx] = (tts_file, silence_file)
                            generation_times.append(gen_time)
                            audio_durations.append(audio_dur)

                            if segment.break_after > 0:
                                pause_duration = segment.break_after
                            elif self.pause_after is not None:
                                pause_duration = self.pause_after
                            else:
                                pause_duration = 0
                            pause_info = f" | +{pause_duration:.1f}s pause" if pause_duration > 0 else ""
                            text_preview = segment.text[:60] + "..." if len(segment.text) > 60 else segment.text
                            print(
                                f"[{completed}/{len(segments)}] "
                                f"ID {segment.segment_id:03d} "
                                f"| gen {gen_time:.2f}s "
                                f"| {text_preview}{pause_info}"
                            )
                        else:
                            print(f"[{completed}/{len(segments)}] Segment {idx + 1} [ID: {segment.segment_id:03d}]: ERROR - {error}")
                            results[idx] = None
                except KeyboardInterrupt:
                    print("\nKeyboard interrupt received. Cancelling pending tasks...")
                    for f in future_to_task:
                        f.cancel()
                    executor.shutdown(cancel_futures=True)
                    raise
                except Exception as e:
                    task = future_to_task.get(future)
                    idx = task["idx"] if task else -1
                    segment = task["segment"] if task else None
                    seg_info = f"Segment {idx + 1}" if idx >= 0 else "Segment"
                    seg_id = f" [ID: {segment.segment_id:03d}]" if segment else ""
                    print(f"[{completed}/{len(segments)}] {seg_info}{seg_id}: FAILED - {e}")
                    results[idx] = None

        # Build file lists for concatenation and cleanup
        print("\nAssembling audio files in correct order...")
        files_for_concat = []  # Files to concatenate (using cached silence)

        for idx, result in enumerate(results):
            if result is not None:
                tts_file, silence_file = result

                # Add TTS file
                if tts_file:
                    files_for_concat.append(tts_file)
                    self.temp_files.append(tts_file)

                # Handle silence files
                    if silence_file:
                        # Always add to temp_files for cleanup
                        self.temp_files.append(silence_file)

                        # For concatenation, use cached version to avoid duplicates
                    # Use segment's break_after if present; otherwise pause_after if set
                    if segments[idx].break_after > 0:
                        pause_duration = segments[idx].break_after
                    elif self.pause_after is not None:
                        pause_duration = self.pause_after
                    else:
                        pause_duration = 0

                    # Check if we have a valid cached silence file that still exists
                    if pause_duration in self.silence_cache and self.silence_cache[pause_duration].exists():
                        files_for_concat.append(self.silence_cache[pause_duration])
                    else:
                        # Use the newly generated silence file and update cache
                        self.silence_cache[pause_duration] = silence_file
                        files_for_concat.append(silence_file)

        print(f"Total files to concatenate: {len(files_for_concat)}")
        print(f"Total files for cleanup: {len(self.temp_files)}")

        # Print statistics
        if generation_times:
            total_gen_time = sum(generation_times)
            total_audio_time = sum(audio_durations)
            avg_rtf = total_gen_time / total_audio_time if total_audio_time > 0 else 0

            print(f"\n=== Generation Statistics ===")
            print(f"  Total segments processed: {len(generation_times)}")
            print(f"  Total audio generated: {total_audio_time:.1f}s ({total_audio_time/60:.1f} minutes)")
            print(f"  Total generation time: {total_gen_time:.1f}s ({total_gen_time/60:.1f} minutes)")
            print(f"  Average RTF: {avg_rtf:.2f}x")

            if len(generation_times) > 1:
                steady_gen_time = sum(generation_times[1:])
                steady_audio_time = sum(audio_durations[1:])
                steady_rtf = steady_gen_time / steady_audio_time if steady_audio_time > 0 else 0
                print(f"  Steady-state RTF (excl. warmup): {steady_rtf:.2f}x")

        return files_for_concat

    def concatenate_wav_files(self, file_list: List[Path], output_path: Path):
        """
        Concatenate multiple WAV files into a single file.
        """
        if not file_list:
            print("No files to concatenate")
            return

        print(f"\n=== PHASE 3: Concatenation ===")
        print(f"Concatenating {len(file_list)} files...")

        # Remove existing output file if it exists
        if output_path.exists():
            output_path.unlink()
            print(f"Removed existing file: {output_path.name}")

        # Read the first file to get parameters
        with wave.open(str(file_list[0]), 'rb') as first_wav:
            params = first_wav.getparams()
            frames = first_wav.readframes(first_wav.getnframes())

        all_frames = [frames]

        # Read remaining files
        for filepath in file_list[1:]:
            with wave.open(str(filepath), 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                all_frames.append(frames)

        # Write concatenated file
        with wave.open(str(output_path), 'wb') as output_wav:
            output_wav.setparams(params)
            output_wav.writeframes(b''.join(all_frames))

        print(f"✓ Concatenation complete: {output_path.name}")

    def cleanup_temp_files(self):
        """
        Remove temporary audio files.
        """
        if not self.temp_files:
            print("\nNo temporary files to clean up.")
            return

        print(f"\nCleaning up {len(self.temp_files)} temporary file(s)...")
        unique_files = set(self.temp_files)
        deleted_count = 0
        errors = []

        for filepath in unique_files:
            try:
                if filepath.exists():
                    filepath.unlink()
                    deleted_count += 1
            except Exception as e:
                errors.append(f"  Error deleting {filepath.name}: {e}")

        if errors:
            print("\nCleanup errors:")
            for error in errors:
                print(error)

        print(f"✓ Cleaned up {deleted_count} temporary files")
        self.temp_files = []
        # Clear silence cache since files have been deleted
        self.silence_cache = {}

    def process_text_file(self, input_file: str, output_name: str = None,
                         min_duration: float = 30.0, max_duration: float = 60.0):
        """
        Process a long-form text file with intelligent segmentation.

        Args:
            input_file: Path to input text file
            output_name: Base name for output files
            min_duration: Minimum target segment duration
            max_duration: Maximum target segment duration
        """
        input_path = Path(input_file).expanduser()

        # If relative path, look in default input directory
        if not input_path.is_absolute():
            default_input_dir = get_user_subdir("input")
            input_path = default_input_dir / Path(input_file)
            if not input_path.exists():
                current_dir_path = Path(input_file)
                if current_dir_path.exists():
                    input_path = current_dir_path
                else:
                    print(f"Error: File not found in {default_input_dir} or current directory")
                    return None

        if output_name is None:
            output_name = input_path.stem

        print(f"\n{'='*60}")
        print(f"Long-Form TTS Processing")
        print(f"{'='*60}")
        print(f"Input file: {input_path}")
        print(f"Output base name: {output_name}")

        # Start overall timer
        overall_start_time = time.time()

        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Count total words (excluding tags)
        text_without_tags = re.sub(r'<[^>]+>', '', text)
        total_words = len(text_without_tags.split())

        # Phase 1: Segment text
        contains_tags = ("<voice" in text) or ("<break" in text)
        if contains_tags:
            segments = self.parse_text_with_breaks(input_path)
        else:
            # If pause_after is set, use it for segmentation duration
            if self.pause_after is not None:
                target_duration = self.pause_after
                # Use a very tight range to keep segments short for language learning
                # Target 80-100% of pause_after to account for TTS variability
                segments = self.segment_text(text, target_duration * 0.8, target_duration * 1.0)
            else:
                segments = self.segment_text(text, min_duration, max_duration)

        if not segments:
            print("Error: No segments generated")
            return None

        # Phase 2: Generate audio
        try:
            audio_files = self.generate_audio(segments, output_name)
        except KeyboardInterrupt:
            print("\nAborted during audio generation. Cleaning up temporary files...")
            self.cleanup_temp_files()
            raise

        if not audio_files:
            print("Error: No audio files generated")
            return None

        # Phase 3: Concatenate
        final_output = self.output_dir / f"{output_name}.wav"
        try:
            self.concatenate_wav_files(audio_files, final_output)
        except KeyboardInterrupt:
            print("\nAborted during concatenation.")
            self.cleanup_temp_files()
            raise

        # Calculate total time
        total_time = time.time() - overall_start_time

        print(f"\n{'='*60}")
        print(f"✓ COMPLETE")
        print(f"{'='*60}")
        print(f"Final audio: {final_output}")
        print(f"Total words processed: {total_words}")
        print(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

        return final_output


def main():
    ensure_user_directories()
    parser = argparse.ArgumentParser(
        description="Generate audio from long-form text with intelligent segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s input.txt
  %(prog)s input.txt --min-duration 20 --max-duration 45
  %(prog)s input.txt --device cpu --keep-temp
        """
    )
    parser.add_argument(
        "input_file",
        help="Path to input text file or directory containing text files"
    )
    parser.add_argument(
        "-o", "--output-name",
        help="Base name for output files (default: input filename without extension)"
    )
    parser.add_argument(
        "-d", "--output-dir",
        default=str(get_user_subdir("output")),
        help="Directory for output files (default: ~/Documents/AlmondTTS/output)"
    )
    parser.add_argument(
        "-r", "--reference-audio",
        default=None,
        help="Path to reference audio file for voice cloning (optional, uses built-in speaker if not provided)"
    )
    parser.add_argument(
        "-l", "--language",
        default="es",
        help="Language code (default: es)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda", "auto"],
        default="auto",
        help="Device to use for inference (default: auto)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=30.0,
        help="Minimum target segment duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        help="Maximum target segment duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary audio files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for TTS generation (default: 1)"
    )
    parser.add_argument(
        "--pause-after",
        type=float,
        default=None,
        help="Add a pause of this many seconds after each audio segment (overrides break tags)"
    )
    parser.add_argument(
        "--voice-map",
        type=str,
        default=None,
        help='JSON mapping of language codes to voice files, e.g., \'{"en": "english.wav", "es": null}\''
    )
    parser.add_argument(
        "--auto-detect-language",
        action="store_true",
        help="Automatically detect language per segment and use corresponding voice from voice-map"
    )

    args = parser.parse_args()

    # Parse voice map if provided
    voice_map = None
    if args.voice_map:
        try:
            voice_map = json.loads(args.voice_map)
            # Validate it's a dictionary
            if not isinstance(voice_map, dict):
                print("Error: --voice-map must be a JSON object/dictionary")
                return
        except json.JSONDecodeError as e:
            print(f"Error parsing --voice-map JSON: {e}")
            return

    # Validate input path exists BEFORE initializing the model
    input_path = Path(args.input_file).expanduser()
    if not input_path.is_absolute():
        default_input_dir = get_user_subdir("input")
        input_path = default_input_dir / Path(args.input_file)
        if not input_path.exists():
            # Try current directory as fallback
            current_dir_path = Path(args.input_file)
            if current_dir_path.exists():
                input_path = current_dir_path
            else:
                print(f"\nError: Input path not found: {args.input_file}")
                print(f"\nSearched in:")
                print(f"  1. {default_input_dir / Path(args.input_file)}")
                print(f"  2. {Path.cwd() / args.input_file}")
                print(f"\nPlease check:")
                print(f"  - The filename/directory spelling is correct")
                print(f"  - The file/directory exists in one of the above locations")
                print(f"  - Or provide the full absolute path")
                return
    elif not input_path.exists():
        print(f"\nError: Input path not found: {input_path}")
        print(f"\nPlease check that the path exists.")
        return

    # Check if input is a directory or file
    if input_path.is_dir():
        # Get all text files in the directory
        text_files = sorted([f for f in input_path.glob("*.txt")])

        if not text_files:
            print(f"\nError: No .txt files found in directory: {input_path}")
            return

        print(f"Found {len(text_files)} text file(s) in directory: {input_path}\n")
        for i, f in enumerate(text_files, 1):
            print(f"  {i}. {f.name}")
        print()

        # Initialize processor once for all files
        processor = LongFormTTS(
            speaker_wav=args.reference_audio,
            language=args.language,
            output_dir=args.output_dir,
            device=None if args.device == "auto" else args.device,
            num_workers=args.workers,
            pause_after=args.pause_after,
            voice_map=voice_map,
            auto_detect_language=args.auto_detect_language
        )

        # Process each file
        successful = 0
        failed = 0
        for i, text_file in enumerate(text_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing file {i}/{len(text_files)}: {text_file.name}")
            print(f"{'='*60}\n")

            # Use the file's stem as output name if not specified
            output_name = args.output_name if args.output_name else text_file.stem

            final_output = processor.process_text_file(
                input_file=str(text_file),
                output_name=output_name,
                min_duration=args.min_duration,
                max_duration=args.max_duration
            )

            if final_output:
                successful += 1
            else:
                failed += 1
                print(f"\nWarning: Failed to process {text_file.name}")

            # Cleanup temp files after each file
            if not args.keep_temp:
                processor.cleanup_temp_files()

        # Print summary
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total files: {len(text_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"{'='*60}\n")

    else:
        # Single file processing (original behavior)
        print(f"Input file validated: {input_path}\n")

        # Initialize processor
        processor = LongFormTTS(
            speaker_wav=args.reference_audio,
            language=args.language,
            output_dir=args.output_dir,
            device=None if args.device == "auto" else args.device,
            num_workers=args.workers,
            pause_after=args.pause_after,
            voice_map=voice_map,
            auto_detect_language=args.auto_detect_language
        )

        # Process the file
        final_output = processor.process_text_file(
            input_file=str(input_path),
            output_name=args.output_name,
            min_duration=args.min_duration,
            max_duration=args.max_duration
        )

        # Only cleanup if processing was successful
        if final_output is None:
            print("\nProcessing failed. Exiting.")
            return

        # Cleanup if requested
        if not args.keep_temp:
            processor.cleanup_temp_files()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user (Ctrl+C).")
