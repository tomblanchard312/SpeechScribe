"""Speech engine adapters for the canonical ``src/speechscribe`` package.

Heavy model dependencies are imported lazily so the base package remains usable
without every optional runtime installed.
"""

from __future__ import annotations

import copy
import logging
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .models import AudioFrame, TranscriptSegment

logger = logging.getLogger(__name__)


def _frame_duration_ms(frame: AudioFrame) -> int:
    bytes_per_sample = 2  # SpeechScribe normalizes PCM to signed 16-bit.
    bytes_per_second = frame.sample_rate * frame.channels * bytes_per_sample
    if bytes_per_second <= 0:
        return 0
    return int((len(frame.data) / bytes_per_second) * 1000)


def write_frames_to_wav(frames: Sequence[AudioFrame], path: Path) -> Path:
    """Write normalized PCM frames to one valid WAV file.

    All frames must use the same sample rate and channel count. SpeechScribe's
    canonical AudioFrame payload is 16-bit little-endian PCM.
    """
    if not frames:
        raise ValueError("At least one audio frame is required")

    sample_rate = frames[0].sample_rate
    channels = frames[0].channels
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("Audio frame has invalid sample rate or channel count")

    for frame in frames:
        if frame.sample_rate != sample_rate or frame.channels != channels:
            raise ValueError(
                "Cannot combine frames with different sample rates/channel counts; "
                "normalize them before ASR"
            )

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for frame in frames:
            wf.writeframes(frame.data)

    return path


class WhisperASREngine:
    """faster-whisper ASR backend."""

    name = "whisper"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Whisper backend requires faster-whisper and torch"
            ) from exc

        configured_device = self.config.get("device", "auto")
        if configured_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = configured_device

        compute_type = self.config.get(
            "compute_type", "float16" if device == "cuda" else "int8"
        )
        model_name = self.config.get("model", "small")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        logger.info("Loaded Whisper model %s on %s", model_name, device)

    def transcribe(self, frames: Sequence[AudioFrame]) -> List[TranscriptSegment]:
        if not frames:
            return []

        session_id = frames[0].session_id
        base_offset_ms = frames[0].timestamp_ms

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = Path(tmp.name)

        try:
            write_frames_to_wav(frames, temp_path)
            segments_iter, info = self.model.transcribe(
                str(temp_path),
                language=self.config.get("language"),
                task="translate" if self.config.get("translate", False) else "transcribe",
                vad_filter=self.config.get("vad_filter", True),
            )

            results: List[TranscriptSegment] = []
            for segment in segments_iter:
                results.append(
                    TranscriptSegment(
                        session_id=session_id,
                        start_ms=base_offset_ms + int(segment.start * 1000),
                        end_ms=base_offset_ms + int(segment.end * 1000),
                        text=segment.text.strip(),
                        language=info.language or self.config.get("language") or "unknown",
                        confidence=getattr(segment, "confidence", None),
                        metadata={"engine": self.name},
                    )
                )
            return results
        finally:
            temp_path.unlink(missing_ok=True)


class VibeVoiceASREngine:
    """Microsoft VibeVoice-ASR backend using the Transformers integration.

    The parsed decoder output contains Start, End, Speaker, and Content fields,
    so diarization and timestamps can be preserved directly in TranscriptSegment.
    """

    name = "vibevoice_asr"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_id = config.get("vibevoice_asr_model", "microsoft/VibeVoice-ASR-HF")
        self.processor = None
        self.model = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "VibeVoice ASR requires a Transformers release with VibeVoice ASR "
                "support plus torch"
            ) from exc

        configured_device = self.config.get("device", "auto")
        if configured_device == "auto":
            if torch.cuda.is_available():
                device_map = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device_map = "mps"
            else:
                device_map = "cpu"
        else:
            device_map = configured_device

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=device_map,
        )
        self.model.eval()
        logger.info("Loaded VibeVoice ASR model %s on %s", self.model_id, device_map)

    def transcribe(self, frames: Sequence[AudioFrame]) -> List[TranscriptSegment]:
        if not frames:
            return []

        session_id = frames[0].session_id
        base_offset_ms = frames[0].timestamp_ms

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = Path(tmp.name)

        try:
            write_frames_to_wav(frames, temp_path)
            prompt = self.config.get("context_prompt") or self.config.get("hotwords")
            kwargs: Dict[str, Any] = {"audio": str(temp_path)}
            if prompt:
                kwargs["prompt"] = prompt

            inputs = self.processor.apply_transcription_request(**kwargs)
            inputs = inputs.to(self.model.device, self.model.dtype)

            generation_kwargs = {
                "max_new_tokens": self.config.get("max_new_tokens", 2048)
            }
            output_ids = self.model.generate(**inputs, **generation_kwargs)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            parsed = self.processor.decode(generated_ids, return_format="parsed")[0]

            language = self.config.get("language") or "auto"
            results: List[TranscriptSegment] = []
            for item in parsed:
                start_s = float(item.get("Start", 0.0))
                end_s = float(item.get("End", start_s))
                speaker = item.get("Speaker")
                text = str(item.get("Content", "")).strip()
                if not text:
                    continue

                speaker_id = None if speaker is None else f"speaker_{speaker}"
                results.append(
                    TranscriptSegment(
                        session_id=session_id,
                        start_ms=base_offset_ms + int(start_s * 1000),
                        end_ms=base_offset_ms + int(end_s * 1000),
                        text=text,
                        language=language,
                        speaker_id=speaker_id,
                        speaker_label=None if speaker is None else f"Speaker {speaker}",
                        metadata={
                            "engine": self.name,
                            "model": self.model_id,
                            "native_diarization": True,
                        },
                    )
                )
            return results
        finally:
            temp_path.unlink(missing_ok=True)


class VibeVoiceRealtimeTTSEngine:
    """Microsoft VibeVoice-Realtime-0.5B single-speaker TTS backend.

    Upstream realtime TTS uses cached voice prompt files. SpeechScribe therefore
    requires ``vibevoice_voice_preset`` to point at an approved upstream .pt
    voice preset rather than attempting arbitrary voice cloning.
    """

    name = "vibevoice_realtime"
    sample_rate = 24000

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_id = config.get(
            "vibevoice_tts_model", "microsoft/VibeVoice-Realtime-0.5B"
        )
        preset = config.get("vibevoice_voice_preset")
        if not preset:
            raise ValueError(
                "VibeVoice realtime TTS requires 'vibevoice_voice_preset' to "
                "reference an upstream cached .pt voice preset"
            )
        self.voice_preset = Path(preset)
        if not self.voice_preset.exists():
            raise FileNotFoundError(self.voice_preset)

        self.processor = None
        self.model = None
        self.device = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_streaming_processor import (
                VibeVoiceStreamingProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "VibeVoice realtime TTS requires the microsoft/VibeVoice package "
                "installed with its streamingtts extras"
            ) from exc

        configured = self.config.get("device", "auto")
        if configured == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = configured

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        attn = "flash_attention_2" if device == "cuda" else "sdpa"
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_id)
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=device if device in ("cuda", "cpu") else None,
                attn_implementation=attn,
            )
        except Exception:
            if attn != "flash_attention_2":
                raise
            logger.warning("FlashAttention load failed; retrying VibeVoice TTS with SDPA")
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=device if device in ("cuda", "cpu") else None,
                attn_implementation="sdpa",
            )

        if device == "mps":
            self.model.to("mps")
        self.model.eval()
        self.model.set_ddpm_inference_steps(
            num_steps=int(self.config.get("vibevoice_inference_steps", 5))
        )
        self.device = device

    def synthesize(self, text: str, output_path: Path) -> Path:
        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        target_device = self.device if self.device != "cpu" else "cpu"
        with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
            cached_prompt = torch.load(
                self.voice_preset,
                map_location=target_device,
                weights_only=True,
            )

        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=cached_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(target_device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=float(self.config.get("vibevoice_cfg_scale", 1.5)),
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            all_prefilled_outputs=copy.deepcopy(cached_prompt),
        )
        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            raise RuntimeError("VibeVoice realtime TTS produced no audio")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.processor.save_audio(outputs.speech_outputs[0], output_path=str(output_path))
        return output_path


def create_asr_engine(name: str, config: Dict[str, Any]):
    normalized = name.lower()
    if normalized == "whisper":
        return WhisperASREngine(config)
    if normalized in {"vibevoice", "vibevoice_asr"}:
        return VibeVoiceASREngine(config)
    raise NotImplementedError(f"ASR engine '{name}' is not implemented")


def audio_frames_duration_ms(frames: Sequence[AudioFrame]) -> int:
    """Return total PCM duration for a normalized frame sequence."""
    return sum(_frame_duration_ms(frame) for frame in frames)
