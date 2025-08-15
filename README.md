# SpeechScribe - Comprehensive Speech Processing Tool

A powerful, offline-capable speech processing tool built with OpenAI's Whisper technology and advanced TTS capabilities. Perfect for transcription, voice synthesis, voice cloning, and voice training without requiring an internet connection.

## ✨ Features

- **Fully Offline**: No internet connection required after initial setup
- **Multiple Audio Formats**: Supports `.m4a`, `.mp3`, `.wav`, `.mov`, `.mp4`, and many other formats
- **Multiple Output Formats**: Generates plain text, SRT subtitles, and VTT captions
- **Language Detection**: Automatically detects the language of speech
- **Translation Support**: Optionally translate non-English speech to English
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **GPU Acceleration**: Optional CUDA support for faster processing
- **Smart Audio Processing**: Automatic conversion to optimal format for Whisper
- **🎤 Voice Synthesis**: Text-to-speech with multiple engines (Coqui TTS, ElevenLabs, Azure)
- **🎭 Voice Cloning**: Clone voices from audio samples with YourTTS
- **🔄 Voice Conversion**: Convert one voice to another
- **🎨 Voice Training**: Train custom voice models from audio samples
- **🌿 Enhanced Naturalness**: Advanced voice quality improvements and naturalness features

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (optional but recommended for audio conversion)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tomblanchard312/SpeechScribe.git
   cd SpeechScribe
   ```

2. **Install in development mode:**
   ```bash
   pip install -e .
   ```

   Or install dependencies manually:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg (recommended):**
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use Chocolatey: `choco install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)

## 📖 Usage

### Command Line Interface

SpeechScribe provides a powerful CLI with multiple commands:

```bash
# Transcribe a single file
speechscribe transcribe audio_file.mp3

# Process multiple files in batch
speechscribe batch /path/to/audio/directory

# Show file information
speechscribe info audio_file.mp3

# View configuration
speechscribe config

# Reset configuration to defaults
speechscribe reset-config

# Voice synthesis and manipulation
speechscribe speak "Hello world" --output hello.wav
speechscribe clone-voice source.mp3 "New text to speak" --output cloned.wav
speechscribe convert-voice source.mp3 target.mp3 --output converted.wav
speechscribe voices --engine coqui_tts
speechscribe voice-engines

# Voice training and management
speechscribe train-voice /path/to/audio/folder --output voice_name --quality high
speechscribe train-voice-enhanced /path/to/audio/folder voice_name --quality ultra
speechscribe trained-voices
speechscribe models

# Enhanced and expressive voice synthesis
speechscribe enhanced-speak "Enhanced natural speech" --output enhanced.wav --quality high --emotion Happy
speechscribe expressive-speak "Truly expressive and natural speech!" --output expressive.wav --emotion Surprised --emphasis strong

# Speed control options
speechscribe speak "Hello world" --output hello.wav --no-fix-speed
speechscribe clone-voice source.mp3 "New text" --output cloned.wav --no-fix-speed
```

### Basic Transcription

```bash
# Simple transcription
speechscribe transcribe audio_file.mp3

# With custom options
speechscribe transcribe audio_file.mp3 --model large-v3 --device cuda --translate
```

This will create multiple output files based on your configuration:
- `audio_file_transcript.txt` - Plain text transcript
- `audio_file_transcript.srt` - SRT subtitle file  
- `audio_file_transcript.vtt` - VTT caption file
- `audio_file_transcript.json` - Structured data with metadata
- `audio_file_transcript.csv` - Spreadsheet format
- `audio_file_transcript.md` - Markdown with timestamps

### Advanced Options

```bash
# Use a larger model for better accuracy
speechscribe transcribe audio_file.mp3 --model large-v3

# Use GPU acceleration (if available)
speechscribe transcribe audio_file.mp3 --device cuda

# Translate non-English speech to English
speechscribe transcribe audio_file.mp3 --translate

# Force a specific language
speechscribe transcribe audio_file.mp3 --language es

# Skip audio conversion (use original file format)
speechscribe transcribe audio_file.mp3 --no-convert

# Custom output formats
speechscribe transcribe audio_file.mp3 --formats txt json csv

# Custom output directory
speechscribe transcribe audio_file.mp3 --output-dir /path/to/output
```

### Batch Processing

```bash
# Process all audio files in a directory
speechscribe batch /path/to/audio/directory

# Process with custom options
speechscribe batch /path/to/audio/directory --model medium --recursive

# Custom file patterns
speechscribe batch /path/to/audio/directory --pattern "*.mp3,*.m4a"

# Output to specific directory
speechscribe batch /path/to/audio/directory --output-dir /path/to/output
```

### Voice Training

```bash
# Train a voice model from a folder of audio files
speechscribe train-voice /path/to/audio/folder --output voice_name --quality high

# Train with ULTRA quality for maximum realism
speechscribe train-voice /path/to/audio/folder --output voice_name --quality ultra

# Train with custom duration filters
speechscribe train-voice /path/to/audio/folder --output voice_name --min-duration 5.0 --max-duration 300.0

# Process subdirectories recursively
speechscribe train-voice /path/to/audio/folder --output voice_name --recursive

# Custom file patterns
speechscribe train-voice /path/to/audio/folder --output voice_name --pattern "*.mp3,*.wav,*.m4a"

# List trained voices
speechscribe trained-voices

# Use trained voice for speech
speechscribe speak "Hello world!" --output hello.wav --voice voice_name

# Analyze audio quality before training
speechscribe analyze-audio /path/to/audio/folder --min-score 0.6
```

**Voice Training Benefits:**
- **Reusable Models**: Train once, use many times
- **Quality Control**: Filter audio files by duration and format
- **Batch Processing**: Process multiple audio files automatically
- **Persistent Storage**: Voice models are saved for future use
- **Easy Integration**: Use trained voices with the `speak` command

**Quality Levels:**
- **Low**: Basic voice synthesis (16kHz, 16-bit)
- **Medium**: Enhanced voice synthesis (32kHz, 16-bit)
- **High**: Professional voice synthesis (44.1kHz, 24-bit) ⭐ **Recommended**
- **Ultra**: Maximum realism (48kHz, 24-bit, studio processing) 🚀 **Best Quality**

**Duration Filtering:**
- **Default Range**: 5.0s - 300.0s (5 seconds to 5 minutes)
- **Customizable**: Use `--min-duration` and `--max-duration` options
- **Quality Control**: Longer audio files (10s+) provide better training results

**Audio Quality Analysis:**
Use the `analyze-audio` command to assess your audio files before training:
- **Quality Scoring**: 0.0 to 1.0 scale based on sample rate, bit depth, duration, and codec
- **File Filtering**: Set minimum quality thresholds to focus on the best files
- **Training Recommendations**: Get suggestions for optimal quality settings
- **Detailed Reports**: See individual file scores and improvement suggestions

### Enhanced Voice Synthesis

For maximum naturalness and inflection, use the `enhanced-speak` command:

```bash
# Basic enhanced synthesis
speechscribe enhanced-speak "Hello world!" --output enhanced.wav --quality high

# With emotion and emphasis
speechscribe enhanced-speak "This is amazing!" --output excited.wav --quality high --emotion Happy --emphasis strong

# Custom speed and pitch
speechscribe enhanced-speak "Slow and deep voice" --output deep.wav --quality high --emotion Sad --speed 0.8 --pitch -2
```

**Enhanced Features:**
- **Advanced Text Preprocessing**: Natural prosody, emotional expression, breathing patterns
- **Quality Control**: Low, medium, and high quality settings with different feature levels
- **Emotional Expression**: Happy, Sad, Angry, Fearful, Disgusted, Surprised, Neutral
- **Prosody Enhancement**: Natural pauses, emphasis control, pitch variation
- **Advanced Models**: VITS, FastPitch with HiFiGAN vocoder for maximum naturalness

### Expressive Speech Synthesis

For truly natural, emotionally expressive speech that sounds completely human, use the `expressive-speak` command:

```bash
# Create expressive happy speech
speechscribe expressive-speak "This is absolutely incredible! I can't believe how natural this sounds!" --output happy.wav --emotion Happy --emphasis strong --speed 1.1 --pitch 2

# Create contemplative, serious speech
speechscribe expressive-speak "This is a moment for deep reflection and thoughtful consideration." --output serious.wav --emotion Sad --emphasis weak --speed 0.8 --pitch -1

# Create surprised, excited speech
speechscribe expressive-speak "Wow! This is amazing! The voice synthesis is incredible!" --output surprised.wav --emotion Surprised --emphasis strong --speed 1.2 --pitch 3
```

**Expressive Speech Features:**
- **Maximum Naturalness**: Uses advanced SSML markup and natural speech patterns
- **Emotional Intelligence**: Automatically detects and enhances emotional content
- **Natural Prosody**: Breathing patterns, natural pauses, and rhythm variation
- **Advanced Models**: FastPitch with HiFiGAN vocoder for perfect inflection
- **Quality Settings**: High-quality parameters for studio-grade output
- **Voice Compatibility**: Works with both default voices and trained voice models

**Emotion Options:**

**For `speak`, `clone-voice`, and `enhanced-speak` commands:**
- **Happy**: Bright, cheerful, upbeat with rising pitch and faster rate
- **Sad**: Melancholy, soft, gentle with lower pitch and slower rate
- **Angry**: Forceful, intense, strong with high pitch and fast rate
- **Fearful**: Whisper, nervous, trembling with soft volume and slow rate
- **Disgusted**: Sarcastic, dry, flat with lower pitch and slow rate
- **Surprised**: Excited, amazed, wonder with very high pitch and fast rate

**For `expressive-speak` command (includes all above +):**
- **Neutral**: Balanced, natural, conversational with normal parameters

## 🔧 Configuration

SpeechScribe uses a YAML configuration file for persistent settings. The configuration file is automatically created on first run.

### Configuration Commands

```bash
# View current configuration
speechscribe config

# Reset to default configuration
speechscribe reset-config
```

### Configuration Options

The configuration file includes settings for:
- **Model Selection**: Whisper model size (tiny, base, small, medium, large-v3)
- **Device**: CPU or CUDA for inference
- **Output Formats**: Which formats to generate (txt, srt, vtt, json, csv, md)
- **Translation**: Whether to translate non-English speech
- **Voice Synthesis**: Engine preferences and quality settings

## 📚 Examples

### Basic Usage

```python
from speechscribe import transcribe_audio, batch_transcribe, Config

# Single file transcription
segments, metadata = transcribe_audio(
    "audio_file.mp3",
    model="small",
    device="cpu"
)

# Batch processing
results = batch_transcribe(
    ["file1.mp3", "file2.wav"],
    model="medium",
    device="cuda"
)

# Configuration management
config = Config()
config.set('model', 'large-v3')
config.set('device', 'cuda')
```

See the `examples/` directory for more detailed usage examples.

## 🛠️ Development

### Project Structure

```
SpeechScribe/
├── src/speechscribe/
│   ├── __init__.py          # Package initialization
│   ├── cli.py              # Command line interface
│   ├── core.py             # Core transcription logic
│   ├── audio.py            # Audio processing utilities
│   ├── output.py           # Output format handlers
│   ├── voice_synthesis.py  # TTS and voice cloning
│   └── config.py           # Configuration management
├── examples/                # Usage examples
├── requirements.txt         # Python dependencies
├── setup.py                # Package configuration
└── README.md               # This file
```

### Dependencies

**Core Dependencies:**
- `faster-whisper>=0.10.0` - Fast Whisper implementation
- `torch>=2.0.0` - PyTorch for ML operations
- `torchaudio>=2.0.0` - Audio processing with PyTorch
- `click>=8.0.0` - Command line interface framework

**Voice Synthesis Dependencies:**
- `TTS>=0.22.0` - Coqui TTS engine
- `elevenlabs>=0.2.26` - ElevenLabs API integration
- `azure-cognitiveservices-speech>=1.31.0` - Azure Speech Services

**Audio Processing Dependencies:**
- `librosa>=0.10.0` - Audio analysis and processing
- `soundfile>=0.12.0` - Audio file I/O

### Development Setup

1. **Clone and install in development mode:**
   ```bash
   git clone https://github.com/tomblanchard312/SpeechScribe.git
   cd SpeechScribe
   pip install -e .
   ```

2. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run tests:**
   ```bash
   pytest
   ```

4. **Code formatting:**
   ```bash
   black src/
   flake8 src/
   mypy src/
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**Audio Conversion Errors:**
- Ensure FFmpeg is installed and accessible in your PATH
- Check that audio files are not corrupted
- Verify supported audio formats

**GPU Acceleration Issues:**
- Ensure CUDA is properly installed
- Check PyTorch CUDA compatibility
- Verify GPU memory availability

**Voice Training Problems:**
- Use high-quality audio files (44.1kHz, 16-bit or higher)
- Ensure audio files are 5-300 seconds in duration
- Check available disk space for model storage

### Getting Help

If you encounter issues:
1. Check the logs in `speechscribe.log`
2. Review the configuration with `speechscribe config`
3. Try running with verbose logging: `speechscribe --verbose transcribe file.mp3`
4. Open an issue on GitHub with detailed error information

## 📈 Roadmap

- [ ] **Real-time Transcription**: Live audio streaming support
- [ ] **Multi-language Models**: Support for more languages
- [ ] **Advanced Voice Editing**: Pitch, speed, and style controls
- [ ] **Batch Voice Training**: Multiple voice models simultaneously
- [ ] **Web Interface**: Browser-based GUI
- [ ] **API Server**: REST API for integration
- [ ] **Mobile Support**: iOS and Android applications

## 🙏 Acknowledgments

- **OpenAI Whisper**: For the excellent speech recognition technology
- **Coqui TTS**: For the open-source text-to-speech engine
- **YourTTS**: For voice cloning capabilities
- **FFmpeg**: For audio format conversion support

---

**SpeechScribe** - Making speech processing accessible, powerful, and offline-capable. 🎤✨