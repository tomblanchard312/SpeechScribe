# SpeechScribe Project Scope

SpeechScribe is the speech-intelligence engine. Its responsibility ends at producing speech-derived artifacts and optional synthesized audio from supported media inputs.

## In scope

- Audio and video ingestion needed to obtain speech input
- Audio normalization and preprocessing
- Automatic speech recognition (ASR)
- Speaker diarization and speaker metadata
- Language detection and translation
- Transcript post-processing and summarization
- Subtitle and caption generation (SRT, VTT, TTML, JSON, text, etc.)
- Live caption output
- Text-to-speech, voice synthesis, and related speech features
- Hardware/runtime selection for supported speech workloads (CPU, CUDA, and future accelerator backends)
- Offline, sovereign, local, and Azure-hosted speech-processing scenarios

## Out of scope

The following belong in a separate media-publishing/orchestration application rather than SpeechScribe:

- Watching OneDrive or other cloud folders for content to publish
- Managing creator publishing queues and publishing schedules
- Generating platform-specific titles, descriptions, tags, or channel metadata
- Creating or composing thumbnails for publishing workflows
- Burning captions into a final distribution copy when the purpose is platform publishing
- Uploading or publishing content to YouTube, Rumble, or other social/video platforms
- Managing YouTube/Rumble OAuth credentials, channel configuration, upload retries, or publish status
- Cross-platform publishing dashboards and creator workflow state

## Integration boundary

A publishing/orchestration application should treat SpeechScribe as a dependency or service.

Typical flow:

```text
Media source / OneDrive
        |
        v
Publishing orchestrator
        |
        +--> SpeechScribe
        |      - extract/normalize speech audio
        |      - transcribe
        |      - diarize
        |      - translate/summarize
        |      - produce transcript + subtitle artifacts
        |
        v
Publishing-specific processing
        - thumbnail composition
        - title/description metadata
        - final video render/caption burn-in when required
        - YouTube/Rumble upload
        - retry/status tracking
```

SpeechScribe should not import platform-specific publisher SDKs. The publishing application may import SpeechScribe as a Python package, invoke its CLI, or call a future SpeechScribe service API.

## Design rule

If a feature remains useful when YouTube, Rumble, OneDrive, and all social-publishing services are removed, it probably belongs in SpeechScribe.

If a feature exists primarily to move completed media through a creator/publishing workflow, it belongs in the publishing application.
