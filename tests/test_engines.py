from pathlib import Path

import pytest

from speechscribe.delivery import FileOutputAdapter
from speechscribe.engines import audio_frames_duration_ms, write_frames_to_wav
from speechscribe.models import AudioFrame


def make_frame(data: bytes, sample_rate: int = 16000, channels: int = 1):
    return AudioFrame(
        session_id="session",
        stream_id="stream",
        timestamp_ms=0,
        data=data,
        sample_rate=sample_rate,
        channels=channels,
    )


def test_audio_frame_duration_uses_pcm_size():
    # 1 second of 16-bit mono PCM at 16 kHz.
    frame = make_frame(b"\x00\x00" * 16000)
    assert audio_frames_duration_ms([frame]) == 1000


def test_write_frames_to_wav_combines_all_pcm(tmp_path: Path):
    first = make_frame(b"\x01\x00" * 8000)
    second = make_frame(b"\x02\x00" * 8000)
    target = tmp_path / "combined.wav"

    write_frames_to_wav([first, second], target)

    import wave

    with wave.open(str(target), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getnframes() == 16000


def test_write_frames_to_wav_rejects_inconsistent_format(tmp_path: Path):
    first = make_frame(b"\x00\x00" * 100)
    second = make_frame(b"\x00\x00" * 100, sample_rate=8000)
    with pytest.raises(ValueError):
        write_frames_to_wav([first, second], tmp_path / "bad.wav")


def test_srt_timestamp_formatting():
    assert FileOutputAdapter._format_timestamp(3_723_004, ",") == "01:02:03,004"
    assert FileOutputAdapter._format_timestamp(3_723_004, ".") == "01:02:03.004"
