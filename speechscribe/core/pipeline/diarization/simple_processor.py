"""
Diarization Module - Simple Processor

Simple rule-based speaker diarization.
"""

import logging
from collections import defaultdict
from typing import List

from ...models.transcript import TranscriptSegment
from .base import DiarizationConfig, DiarizationProcessor

logger = logging.getLogger(__name__)


class SimpleDiarizationProcessor(DiarizationProcessor):
    """
    Simple rule-based speaker diarization.

    Assigns speakers based on time gaps and segment clustering.
    This is a basic implementation for offline/local environments.
    """

    def __init__(self, config: DiarizationConfig):
        super().__init__(config)
        self.speaker_gap_threshold = 2.0  # seconds

    def process(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """
        Assign speaker IDs based on time gaps.

        Simple algorithm: if gap between segments > threshold, assume different speaker.
        """
        if not segments:
            return segments

        logger.info(f"Starting simple diarization on {len(segments)} segments")

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_time)

        # Assign speakers
        current_speaker = 0
        last_end_time = sorted_segments[0].start_time

        for segment in sorted_segments:
            # Check if there's a significant gap
            if segment.start_time - last_end_time > self.speaker_gap_threshold:
                current_speaker += 1

            segment.speaker_id = f"speaker_{current_speaker}"
            last_end_time = max(last_end_time, segment.end_time)

        # Re-cluster speakers if we have too many
        if self.config.max_speakers and current_speaker + 1 > self.config.max_speakers:
            clustered_segments = self._cluster_speakers(
                sorted_segments, self.config.max_speakers
            )
            logger.info(
                f"Clustered {current_speaker + 1} speakers into {self.config.max_speakers}"
            )
            return clustered_segments

        logger.info(f"Diarization completed: {current_speaker + 1} speakers detected")
        return sorted_segments

    def _cluster_speakers(
        self, segments: List[TranscriptSegment], max_speakers: int
    ) -> List[TranscriptSegment]:
        """Cluster speakers when we have too many."""
        # Simple clustering: merge speakers with least time between them
        # This is a basic implementation - real diarization would use ML models

        # Group segments by current speaker
        speaker_groups = defaultdict(list)
        for segment in segments:
            speaker_groups[segment.speaker_id].append(segment)

        # Sort speakers by total duration
        speaker_durations = {}
        for speaker_id, segs in speaker_groups.items():
            duration = sum(s.end_time - s.start_time for s in segs)
            speaker_durations[speaker_id] = duration

        # Keep top speakers by duration, merge others
        sorted_speakers = sorted(
            speaker_durations.items(), key=lambda x: x[1], reverse=True
        )
        keep_speakers = [s[0] for s in sorted_speakers[:max_speakers]]
        merge_speaker = keep_speakers[0] if keep_speakers else "speaker_0"

        # Reassign speakers
        for segment in segments:
            if segment.speaker_id not in keep_speakers:
                segment.speaker_id = merge_speaker

        return segments

    def get_supported_engines(self) -> List[str]:
        """Get list of supported diarization engines."""
        return ["simple"]
