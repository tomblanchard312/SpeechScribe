"""
Shared data models for SpeechScribe platform.

This module defines the canonical data structures used across all layers
of the SpeechScribe platform, ensuring consistency between ingestion,
processing, and delivery.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TranscriptSegment:
    """
    Canonical transcript segment data model.

    This schema normalizes all outputs from speech processing pipelines,
    regardless of whether they come from streaming or batch processing,
    cloud or local engines.
    """

    session_id: str
    start_ms: int
    end_ms: int
    text: str
    language: str
    speaker_id: Optional[str] = None
    speaker_label: Optional[str] = None
    confidence: Optional[float] = None
    translations: Dict[str, str] = field(
        default_factory=dict
    )  # lang_code -> translated_text
    redaction_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "speaker_label": self.speaker_label,
            "confidence": self.confidence,
            "translations": self.translations,
            "redaction_flags": self.redaction_flags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptSegment":
        """Create from dictionary representation."""
        return cls(
            session_id=data["session_id"],
            start_ms=data["start_ms"],
            end_ms=data["end_ms"],
            text=data["text"],
            language=data["language"],
            speaker_id=data.get("speaker_id"),
            speaker_label=data.get("speaker_label"),
            confidence=data.get("confidence"),
            translations=data.get("translations", {}),
            redaction_flags=data.get("redaction_flags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AudioFrame:
    """
    Normalized audio frame data from ingestion adapters.

    All audio sources are normalized to this format before
    entering the speech processing pipeline.
    """

    session_id: str
    stream_id: str
    timestamp_ms: int
    data: bytes  # PCM audio data
    sample_rate: int
    channels: int
    speaker_hint: Optional[str] = None  # Optional speaker identification hint

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (data as base64 for serialization)."""
        import base64

        return {
            "session_id": self.session_id,
            "stream_id": self.stream_id,
            "timestamp_ms": self.timestamp_ms,
            "data": base64.b64encode(self.data).decode("ascii"),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "speaker_hint": self.speaker_hint,
        }


@dataclass
class ProcessingResult:
    """
    Result from speech processing pipeline stages.
    """

    stage_name: str
    segments: List[TranscriptSegment]
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    processing_time_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "segments": [s.to_dict() for s in self.segments],
            "metadata": self.metadata,
            "errors": self.errors,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class SessionMetadata:
    """
    Metadata for a transcription session.
    """

    session_id: str
    created_at: datetime
    source_type: str  # 'file', 'teams', 'zoom', etc.
    profile_name: str
    engine_name: str
    duration_ms: Optional[int] = None
    total_segments: int = 0
    languages_detected: List[str] = field(default_factory=list)
    speakers_identified: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "source_type": self.source_type,
            "profile_name": self.profile_name,
            "engine_name": self.engine_name,
            "duration_ms": self.duration_ms,
            "total_segments": self.total_segments,
            "languages_detected": self.languages_detected,
            "speakers_identified": self.speakers_identified,
        }
