#!/usr/bin/env python3
"""
Basic usage example for VMTranscriber.

This script demonstrates how to use the VMTranscriber package programmatically.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import vmtranscriber
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vmtranscriber import transcribe_audio, batch_transcribe, Config

def main():
    """Demonstrate basic usage."""
    print("🎵 VMTranscriber Basic Usage Example")
    print("=" * 50)
    
    # Example 1: Single file transcription
    print("\n1️⃣ Single File Transcription")
    print("-" * 30)
    
    # You would replace this with an actual audio file path
    audio_file = "path/to/your/audio.mp3"
    
    if Path(audio_file).exists():
        try:
            print(f"Transcribing: {audio_file}")
            segments, metadata = transcribe_audio(
                audio_file,
                model="small",
                device="cpu"
            )
            
            print(f"✅ Transcription completed!")
            print(f"📊 Segments: {len(segments)}")
            print(f"🌍 Language: {metadata.get('language', 'unknown')}")
            print(f"⏱️  Duration: {metadata.get('duration', 0):.2f}s")
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
    else:
        print(f"⚠️  Audio file not found: {audio_file}")
        print("   Please update the audio_file path in this script")
    
    # Example 2: Configuration management
    print("\n2️⃣ Configuration Management")
    print("-" * 30)
    
    config = Config()
    print(f"📁 Config file: {config.config_path}")
    print(f"🔧 Current model: {config.get('model')}")
    print(f"💻 Current device: {config.get('device')}")
    
    # Example 3: Batch processing
    print("\n3️⃣ Batch Processing")
    print("-" * 30)
    
    # You would replace this with actual audio file paths
    audio_files = [
        "path/to/audio1.mp3",
        "path/to/audio2.wav",
        "path/to/audio3.m4a"
    ]
    
    existing_files = [f for f in audio_files if Path(f).exists()]
    
    if existing_files:
        print(f"Found {len(existing_files)} existing audio files")
        print("To process them in batch, uncomment the following code:")
        print("""
        try:
            results = batch_transcribe(
                existing_files,
                model="small",
                device="cpu"
            )
            print(f"Batch processing completed: {len(results)} files processed")
        except Exception as e:
            print(f"Batch processing failed: {e}")
        """)
    else:
        print("No audio files found for batch processing example")
    
    print("\n🎉 Example completed!")
    print("\n💡 To use VMTranscriber:")
    print("   • Install: pip install -e .")
    print("   • CLI: vmtranscriber transcribe audio.mp3")
    print("   • Batch: vmtranscriber batch /path/to/audio/directory")
    print("   • Info: vmtranscriber info audio.mp3")

if __name__ == "__main__":
    main()
