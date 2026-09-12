"""Composable speech-processing pipeline."""

import abc
import logging
import time
from typing import Any, Dict, List

from .engines import create_asr_engine
from .models import AudioFrame, ProcessingResult, TranscriptSegment

logger = logging.getLogger(__name__)


class PipelineStage(abc.ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abc.abstractmethod
    def process(self, input_data: Any) -> ProcessingResult:
        pass

    def _create_result(
        self,
        segments: List[TranscriptSegment],
        metadata: Dict[str, Any] = None,
        errors: List[str] = None,
        processing_time_ms: int = None,
    ) -> ProcessingResult:
        return ProcessingResult(
            stage_name=self.name,
            segments=segments,
            metadata=metadata or {},
            errors=errors or [],
            processing_time_ms=processing_time_ms,
        )


class AudioNormalizationStage(PipelineStage):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("audio_normalization", config)

    def process(self, audio_frames: List[AudioFrame]) -> ProcessingResult:
        start_time = time.time()
        errors = []
        for frame in audio_frames:
            if frame.sample_rate != 16000 or frame.channels != 1:
                logger.warning(
                    "Frame %s is %s Hz/%s channel(s); canonical resampling is not yet implemented",
                    frame.stream_id,
                    frame.sample_rate,
                    frame.channels,
                )
        return self._create_result(
            segments=[],
            metadata={"normalized_frames": len(audio_frames)},
            errors=errors,
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


class ASRStage(PipelineStage):
    """Automatic speech recognition via a pluggable engine adapter."""

    def __init__(self, config: Dict[str, Any], engine_name: str):
        super().__init__("asr", config)
        self.engine_name = engine_name
        self.engine = create_asr_engine(engine_name, config)

    def process(self, audio_frames: List[AudioFrame]) -> ProcessingResult:
        start_time = time.time()
        try:
            segments = self.engine.transcribe(audio_frames)
            errors = []
        except Exception as exc:
            logger.exception("ASR engine %s failed", self.engine_name)
            segments = []
            errors = [f"ASR engine {self.engine_name} failed: {exc}"]

        languages = sorted({s.language for s in segments if s.language})
        return self._create_result(
            segments=segments,
            metadata={"engine": self.engine_name, "languages": languages},
            errors=errors,
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


class DiarizationStage(PipelineStage):
    """Diarization fallback stage.

    ASR engines such as VibeVoice can already emit speaker identities. Existing
    native speaker metadata is preserved; this fallback only labels segments
    that arrive without a speaker.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("diarization", config)

    def process(self, segments: List[TranscriptSegment]) -> ProcessingResult:
        start_time = time.time()
        for segment in segments:
            if segment.speaker_id is None:
                segment.speaker_id = "speaker_1"
                segment.speaker_label = "Speaker 1"
                segment.metadata["diarization_fallback"] = True

        speakers = {s.speaker_id for s in segments if s.speaker_id}
        return self._create_result(
            segments=segments,
            metadata={"speakers_identified": len(speakers)},
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


class TranslationStage(PipelineStage):
    def __init__(self, config: Dict[str, Any], target_languages: List[str]):
        super().__init__("translation", config)
        self.target_languages = target_languages

    def process(self, segments: List[TranscriptSegment]) -> ProcessingResult:
        start_time = time.time()
        # Existing translation behavior is retained until a real translation
        # backend is selected. Do not attribute translation to VibeVoice ASR.
        for segment in segments:
            for lang in self.target_languages:
                segment.translations.setdefault(
                    lang, f"[Translation backend not configured]: {segment.text}"
                )
        return self._create_result(
            segments=segments,
            metadata={"target_languages": self.target_languages},
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


class PostProcessingStage(PipelineStage):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("postprocessing", config)

    def process(self, segments: List[TranscriptSegment]) -> ProcessingResult:
        start_time = time.time()
        for segment in segments:
            segment.text = segment.text.strip()
            if segment.text and not segment.text.endswith((".", "!", "?", ",")):
                segment.text += "."
        return self._create_result(
            segments=segments,
            metadata={"postprocessing_applied": True},
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


class SpeechPipeline:
    def __init__(self, profile_name: str, config: Dict[str, Any]):
        self.profile_name = profile_name
        self.config = config
        self.stages: List[PipelineStage] = []
        self._build_pipeline()

    def _build_pipeline(self):
        from .control import RecommendationEngine

        rec = RecommendationEngine().recommend_configuration(self.profile_name)
        profile = rec["profile"]
        engine = self.config.get("asr_engine") or rec["engine"]

        self.stages.append(AudioNormalizationStage(self.config))
        self.stages.append(ASRStage(self.config, engine))
        if profile.diarization_required:
            self.stages.append(DiarizationStage(self.config))
        if profile.translation_required:
            self.stages.append(TranslationStage(self.config, profile.translation_languages))
        self.stages.append(PostProcessingStage(self.config))

        logger.info(
            "Built %s-stage pipeline for %s using ASR engine %s",
            len(self.stages),
            self.profile_name,
            engine,
        )

    def process_audio_frames(self, audio_frames: List[AudioFrame]) -> List[TranscriptSegment]:
        current_data: Any = audio_frames
        for stage in self.stages:
            result = stage.process(current_data)
            if result.errors:
                logger.warning("Stage %s errors: %s", stage.name, result.errors)
            if result.segments:
                current_data = result.segments

        if current_data and isinstance(current_data, list) and isinstance(current_data[0], TranscriptSegment):
            return current_data
        logger.error("Pipeline did not produce transcript segments")
        return []
