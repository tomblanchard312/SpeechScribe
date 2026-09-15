"""
Example: Transcribe and Summarize with Plugins

This example shows how to use the plugin system to:
1. Transcribe audio with Whisper plugin
2. Summarize the transcript with Ollama plugin

Prerequisites:
    1. Install dependencies: pip install faster-whisper
    2. Install Ollama: https://ollama.ai
    3. Pull model: ollama pull qwen2.5
    4. Start Ollama: ollama serve
"""

import sys
from pathlib import Path

# Add workspace root to path so we can import speechscribe.core
sys.path.insert(0, str(Path(__file__).parent.parent))

from speechscribe.core.plugins import get_plugin_loader


def transcribe_and_summarize(audio_file: str):
    """
    Transcribe an audio file and generate a summary.

    Args:
        audio_file: Path to audio file
    """
    print("=" * 70)
    print("Transcribe and Summarize Pipeline")
    print("=" * 70)
    print()

    # Get plugin loader
    loader = get_plugin_loader()

    # Get ASR plugin (Whisper)
    print("1. Loading ASR plugin...")
    asr_plugins = loader.plugins_by_type("asr")
    if not asr_plugins:
        print("Error: No ASR plugin found!")
        print("Make sure speechscribe/plugins/whisper/ exists")
        return

    whisper_desc = asr_plugins[0]
    print(f"   Using: {whisper_desc.name} v{whisper_desc.version}")
    print(f"   Models: {', '.join(whisper_desc.supported_models[:3])}...")
    whisper = whisper_desc.entry_class()
    print()

    # Get Summarization plugin (Ollama)
    print("2. Loading Summarization plugin...")
    summarization_plugins = loader.plugins_by_type("summarization")
    if not summarization_plugins:
        print("Error: No Summarization plugin found!")
        print("Make sure speechscribe/plugins/ollama_llm/ exists")
        return

    ollama_desc = summarization_plugins[0]
    print(f"   Using: {ollama_desc.name} v{ollama_desc.version}")
    print(f"   Models: {', '.join(ollama_desc.supported_models[:3])}...")
    ollama = ollama_desc.entry_class()
    print()

    # Step 1: Transcribe audio
    print("3. Transcribing audio...")
    print(f"   File: {audio_file}")

    try:
        # Use 'base' model for speed (or 'large-v3' for accuracy)
        result = whisper.transcribe(
            audio_file, model="base", language="en"  # Or use detect_language() first
        )

        transcript = result["text"]
        language = result.get("language", "unknown")

        print(f"   Language: {language}")
        print(f"   Length: {len(transcript)} characters")
        print()
        print("   Transcript (first 200 chars):")
        print(f"   {transcript[:200]}...")
        print()

    except Exception as e:
        print(f"   Error: {e}")
        print("   Make sure the audio file exists and faster-whisper is installed")
        return

    # Step 2: Summarize transcript
    print("4. Generating summary...")

    try:
        summary = ollama.summarize(transcript, model="qwen2.5")

        print("   Summary:")
        print("   " + "-" * 66)
        for line in summary.split("\n"):
            print(f"   {line}")
        print("   " + "-" * 66)
        print()

    except Exception as e:
        print(f"   Error: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return

    # Step 3: Generate meeting notes (if transcript looks like a meeting)
    if any(
        keyword in transcript.lower()
        for keyword in ["meeting", "agenda", "action item", "attendee"]
    ):
        print("5. Detected meeting transcript. Generating structured notes...")

        try:
            notes = ollama.generate_meeting_notes(transcript, model="qwen2.5")

            print("   Meeting Notes:")
            print("   " + "-" * 66)
            for line in notes.split("\n"):
                print(f"   {line}")
            print("   " + "-" * 66)
            print()

        except Exception as e:
            print(f"   Error: {e}")
    else:
        print("5. Skipping meeting notes (not a meeting transcript)")
        print()

    print("=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)


def detect_language_example(audio_file: str):
    """
    Detect the language of an audio file.

    Args:
        audio_file: Path to audio file
    """
    print("=" * 70)
    print("Language Detection Example")
    print("=" * 70)
    print()

    # Get ASR plugin
    loader = get_plugin_loader()
    asr_plugins = loader.plugins_by_type("asr")

    if not asr_plugins:
        print("Error: No ASR plugin found!")
        return

    whisper = asr_plugins[0].entry_class()

    print(f"Detecting language: {audio_file}")

    try:
        result = whisper.detect_language(audio_file)
        language = result["language"]
        confidence = result.get("confidence", 0.0)

        print(f"Language: {language}")
        print(f"Confidence: {confidence:.2%}")
        print()

        # Now transcribe with detected language
        print("Transcribing with detected language...")
        transcript_result = whisper.transcribe(audio_file, language=language)
        transcript = transcript_result["text"]

        print(f"Transcript: {transcript[:200]}...")

    except Exception as e:
        print(f"Error: {e}")

    print()


def translate_example(audio_file: str):
    """
    Transcribe and translate non-English audio to English.

    Args:
        audio_file: Path to audio file
    """
    print("=" * 70)
    print("Translation Example")
    print("=" * 70)
    print()

    # Get ASR plugin
    loader = get_plugin_loader()
    asr_plugins = loader.plugins_by_type("asr")

    if not asr_plugins:
        print("Error: No ASR plugin found!")
        return

    whisper = asr_plugins[0].entry_class()

    print(f"Translating to English: {audio_file}")

    try:
        # First detect language
        lang_result = whisper.detect_language(audio_file)
        original_language = lang_result["language"]

        print(f"Original language: {original_language}")

        # Translate to English
        result = whisper.translate(audio_file)
        translated_text = result["text"]

        print(f"Translated text: {translated_text[:200]}...")
        print()

        # Optional: Summarize the translation
        summarization_plugins = loader.plugins_by_type("summarization")
        if summarization_plugins:
            print("Generating summary of translation...")
            ollama = summarization_plugins[0].entry_class()
            summary = ollama.summarize(translated_text, model="qwen2.5")
            print(f"Summary: {summary}")

    except Exception as e:
        print(f"Error: {e}")

    print()


def main():
    """Run examples."""
    print("\n")
    print("🎙️  SpeechScribe Plugin Integration Examples")
    print()

    # Check if audio file provided
    if len(sys.argv) < 2:
        print("Usage: python plugin_integration.py <audio_file>")
        print()
        print("Example:")
        print("  python plugin_integration.py recording.mp3")
        print()
        print("This example will:")
        print("  1. Transcribe the audio using Whisper plugin")
        print("  2. Summarize the transcript using Ollama plugin")
        print("  3. Generate meeting notes (if applicable)")
        print()
        return

    audio_file = sys.argv[1]

    # Check if file exists
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_file}")
        return

    # Run main pipeline
    transcribe_and_summarize(audio_file)

    # Optional: Run other examples
    # detect_language_example(audio_file)
    # translate_example(audio_file)


if __name__ == "__main__":
    main()
