# Microsoft VibeVoice integration

SpeechScribe should support Microsoft VibeVoice as an optional local/open-source speech backend alongside faster-whisper.

## STT / ASR

Preferred model family: `microsoft/VibeVoice-ASR`.

Use cases:

- long-form multilingual transcription
- speaker-aware structured transcription
- timestamps
- user-provided context / hotword-like biasing
- streaming transcription through the VibeVoice streaming checkpoint

SpeechScribe should keep Whisper as a supported backend. VibeVoice should be selected by configuration/profile rather than replacing Whisper globally.

Recommended implementation path:

1. Add a `VibeVoiceASRProcessor` implementing the same SpeechScribe ASR interface as Whisper.
2. Prefer the Hugging Face Transformers integration where practical so SpeechScribe does not need to fork Microsoft model code.
3. Preserve VibeVoice speaker/timestamp output directly in SpeechScribe `TranscriptSegment` data instead of running a redundant diarization pass when model-provided speaker information is available.
4. Keep translation as a separate pipeline stage; do not advertise translation as a VibeVoice-ASR capability unless the selected checkpoint/runtime actually provides it.
5. Support CUDA first, then evaluate XPU/CPU and other runtimes independently.

## TTS

Preferred current model: `microsoft/VibeVoice-Realtime-0.5B`.

Use cases:

- streaming text-to-speech
- low-latency local narration
- long-form speech generation

The older VibeVoice long-form TTS release should not be treated as the primary integration target. The realtime 0.5B model is the current open model to target.

Recommended implementation path:

1. Add a dedicated VibeVoice TTS adapter under the TTS layer.
2. Use the Transformers text-to-speech integration when it exposes the features SpeechScribe needs.
3. Keep TTS engine selection separate from ASR engine selection. The current generic engine registry is not sufficiently modality-aware to safely register a TTS-only engine without risking incorrect ASR recommendations.
4. Add a modality field or separate ASR/TTS registries before registering `VibeVoice-Realtime-0.5B` as a selectable engine.

## Hardware strategy

- NVIDIA CUDA remains the preferred high-performance backend on systems with an RTX GPU.
- CPU fallback remains important for low-resource/offline systems.
- The VibeVoice ASR BitNet runtime can be evaluated separately for efficient CPU transcription.
- NPU support should be treated as an optional future backend rather than a requirement for VibeVoice support.

## Scope

VibeVoice belongs in SpeechScribe because both ASR and TTS are core speech-intelligence capabilities. Publishing, OneDrive automation, thumbnails, and YouTube/Rumble integration remain outside SpeechScribe.
