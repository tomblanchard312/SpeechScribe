"""
Core Speech Pipeline - Composable Processing Stages

This layer contains the platform-agnostic speech processing pipeline.
Stages can be composed to create different processing workflows.

Stages include:
- Audio normalization
- ASR (streaming/batch)
- Diarization
- Translation
- Alignment/Segmentation
- Post-processing
- TTS
"""

import abc
import logging
import time
from typing import Any, Dict, List, Optional

from .models import AudioFrame, ProcessingResult, TranscriptSegment

logger = logging.getLogger(__name__)


class PipelineStage(abc.ABC):
    """
    Base class for pipeline processing stages.

    Stages process AudioFrames or TranscriptSegments and produce
    ProcessingResults with updated segments.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abc.abstractmethod
    def process(self, input_data: Any) -> ProcessingResult:
        """Process input data and return results."""
        pass

    def _create_result(
        self,
        segments: List[TranscriptSegment],
        metadata: Dict[str, Any] = None,
        errors: List[str] = None,
        processing_time_ms: int = None,
    ) -> ProcessingResult:
        """Helper to create ProcessingResult."""
        return ProcessingResult(
            stage_name=self.name,
            segments=segments,
            metadata=metadata or {},
            errors=errors or [],
            processing_time_ms=processing_time_ms,
        )


class AudioNormalizationStage(PipelineStage):
    """
    Audio normalization stage.

    Normalizes audio to consistent format for downstream processing.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("audio_normalization", config)

    def process(self, audio_frames: List[AudioFrame]) -> ProcessingResult:
        """Normalize audio frames."""
        start_time = time.time()

        normalized_frames = []
        errors = []

        for frame in audio_frames:
            try:
                # Ensure 16kHz mono PCM
                if frame.sample_rate != 16000 or frame.channels != 1:
                    # TODO: Implement actual resampling
                    logger.warning(
                        f"Audio normalization not implemented for "
                        f"frame {frame.stream_id}"
                    )
                    normalized_frames.append(frame)
                else:
                    normalized_frames.append(frame)
            except Exception as e:
                errors.append(f"Failed to normalize frame {frame.stream_id}: {e}")

        processing_time = int((time.time() - start_time) * 1000)

        return self._create_result(
            segments=[],  # No segments produced, just normalized audio
            metadata={"normalized_frames": len(normalized_frames)},
            errors=errors,
            processing_time_ms=processing_time,
        )


class ASRStage(PipelineStage):
    """
    Automatic Speech Recognition stage.

    Converts audio to text using configured engine.
    """

    def __init__(self, config: Dict[str, Any], engine_name: str):
        super().__init__("asr", config)
        self.engine_name = engine_name
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the ASR engine."""
        if self.engine_name == "whisper":
            try:
                from faster_whisper import WhisperModel

                model_size = self.config.get("model", "small")
                device = self.config.get("device", "cpu")
                compute_type = "int8" if device == "cpu" else "float16"
                self.engine = WhisperModel(
                    model_size, device=device, compute_type=compute_type
                )
                logger.info(f"Initialized Whisper ASR engine: {model_size}")
            except ImportError:
                raise RuntimeError("faster-whisper not available for Whisper engine")
        else:
            # TODO: Implement other engines
            raise NotImplementedError(f"ASR engine {self.engine_name} not implemented")

    def process(self, audio_frames: List[AudioFrame]) -> ProcessingResult:
        """Transcribe audio frames to text."""
        start_time = time.time()

        segments = []
        errors = []

        for frame in audio_frames:
            try:
                # Convert PCM bytes to format expected by engine
                # For Whisper, we need to save to temp WAV and transcribe
                import io
                import tempfile
                import wave

                # Create in-memory WAV
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(frame.channels)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(frame.sample_rate)
                    wf.writeframes(frame.data)
                wav_buffer.seek(0)

                # Transcribe with Whisper
                segments_iter, info = self.engine.transcribe(
                    wav_buffer,
                    language=self.config.get("language"),
                    vad_filter=self.config.get("vad_filter", True),
                )

                # Convert to TranscriptSegments
                for seg in segments_iter:
                    transcript_seg = TranscriptSegment(
                        session_id=frame.session_id,
                        start_ms=int(seg.start * 1000),
                        end_ms=int(seg.end * 1000),
                        text=seg.text,
                        language=info.language or "unknown",
                        confidence=getattr(seg, "confidence", None),
                        metadata={
                            "engine": self.engine_name,
                            "stream_id": frame.stream_id,
                        },
                    )
                    segments.append(transcript_seg)

            except Exception as e:
                errors.append(f"ASR failed for frame {frame.stream_id}: {e}")

        processing_time = int((time.time() - start_time) * 1000)

        return self._create_result(
            segments=segments,
            metadata={
                "engine": self.engine_name,
                "language": info.language if "info" in locals() else "unknown",
            },
            errors=errors,
            processing_time_ms=processing_time,
        )


class DiarizationStage(PipelineStage):
    """
    Speaker diarization stage.

    Identifies different speakers in audio.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("diarization", config)

    def process(self, segments: List[TranscriptSegment]) -> ProcessingResult:
        """Add speaker identification to segments."""
        start_time = time.time()

        # TODO: Implement actual diarization
        # For now, assign all segments to speaker 1
        for i, segment in enumerate(segments):
            segment.speaker_id = "speaker_1"
            # Simple rotation for demo
            segment.speaker_label = f"Speaker {i % 3 + 1}"

        processing_time = int((time.time() - start_time) * 1000)

        return self._create_result(
            segments=segments,
            metadata={"speakers_identified": 1},  # TODO: actual count
            processing_time_ms=processing_time,
        )


class TranslationStage(PipelineStage):
    """
    Translation stage.

    Translates text to target languages.
    """

    def __init__(self, config: Dict[str, Any], target_languages: List[str]):
        super().__init__("translation", config)
        self.target_languages = target_languages

    def process(self, segments: List[TranscriptSegment]) -> ProcessingResult:
        """Translate segments to target languages."""
        start_time = time.time()

        # TODO: Implement actual translation
        # For now, just copy original text as "translation"
        for segment in segments:
            for lang in self.target_languages:
                segment.translations[lang] = f"[Translated to {lang}]: {segment.text}"

        processing_time = int((time.time() - start_time) * 1000)

        return self._create_result(
            segments=segments,
            metadata={"target_languages": self.target_languages},
            processing_time_ms=processing_time,
        )


class PostProcessingStage(PipelineStage):
    """
    Post-processing stage.

    Applies punctuation, glossary, profanity filtering, etc.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("postprocessing", config)

    def process(self, segments: List[TranscriptSegment]) -> ProcessingResult:
        """Apply post-processing to segments."""
        start_time = time.time()

        # TODO: Implement actual post-processing
        # For now, just basic cleanup
        for segment in segments:
            # Basic punctuation
            if not segment.text.endswith((".", "!", "?", ",")):
                segment.text += "."

        processing_time = int((time.time() - start_time) * 1000)

        return self._create_result(
            segments=segments,
            metadata={"postprocessing_applied": True},
            processing_time_ms=processing_time,
        )


class SpeechPipeline:
    """
    Composable speech processing pipeline.

    Chains together processing stages based on profile requirements.
    """

    def __init__(self, profile_name: str, config: Dict[str, Any]):
        self.profile_name = profile_name
        self.config = config
        self.stages: List[PipelineStage] = []
        self._build_pipeline()

    def _build_pipeline(self):
        """Build pipeline stages based on profile."""
        from .control import RecommendationEngine

        recommender = RecommendationEngine()
        rec_config = recommender.recommend_configuration(self.profile_name)

        profile = rec_config["profile"]
        engine = rec_config["engine"]

        # Always start with audio normalization
        self.stages.append(AudioNormalizationStage(self.config))

        # Add ASR stage
        self.stages.append(ASRStage(self.config, engine))

        # Add optional stages based on profile
        if profile.diarization_required:
            self.stages.append(DiarizationStage(self.config))

        if profile.translation_required:
            self.stages.append(
                TranslationStage(self.config, profile.translation_languages)
            )

        # Always add post-processing
        self.stages.append(PostProcessingStage(self.config))

        logger.info(
            f"Built pipeline with {len(self.stages)} stages for "
            f"profile {self.profile_name}"
        )

    def process_audio_frames(
        self, audio_frames: List[AudioFrame]
    ) -> List[TranscriptSegment]:
        """
        Process audio frames through the pipeline.

        Returns final transcript segments.
        """
        current_data = audio_frames

        for stage in self.stages:
            logger.debug(f"Running stage: {stage.name}")
            result = stage.process(current_data)

            if result.errors:
                logger.warning(f"Stage {stage.name} had errors: {result.errors}")

            # Pass segments to next stage if it produces them
            if result.segments:
                current_data = result.segments
            # Otherwise keep audio frames

        # Final result should be segments
        if (
            isinstance(current_data, list)
            and current_data
            and isinstance(current_data[0], TranscriptSegment)
        ):
            return current_data
        else:
            logger.error("Pipeline did not produce transcript segments")
            return []
