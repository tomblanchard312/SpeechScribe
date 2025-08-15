# VMTranscriber - Offline Voice Mail Transcription

A powerful, offline-capable audio transcription tool built with OpenAI's Whisper technology using the `faster-whisper` library. Perfect for transcribing voice mails, audio recordings, and speech content without requiring an internet connection.

## ✨ Features

- **Fully Offline**: No internet connection required after initial setup
- **Multiple Audio Formats**: Supports `.m4a`, `.mp3`, `.wav`, and many other formats
- **Multiple Output Formats**: Generates plain text, SRT subtitles, and VTT captions
- **Language Detection**: Automatically detects the language of speech
- **Translation Support**: Optionally translate non-English speech to English
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **GPU Acceleration**: Optional CUDA support for faster processing
- **Smart Audio Processing**: Automatic conversion to optimal format for Whisper

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (optional but recommended for audio conversion)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd VMTranscriber
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

VMTranscriber now provides a powerful CLI with multiple commands:

```bash
# Transcribe a single file
vmtranscriber transcribe audio_file.mp3

# Process multiple files in batch
vmtranscriber batch /path/to/audio/directory

# Show file information
vmtranscriber info audio_file.mp3

# View configuration
vmtranscriber config

# Reset configuration to defaults
vmtranscriber reset-config
```

### Basic Transcription

```bash
# Simple transcription
vmtranscriber transcribe audio_file.mp3

# With custom options
vmtranscriber transcribe audio_file.mp3 --model large-v3 --device cuda --translate
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
vmtranscriber transcribe audio_file.mp3 --model large-v3

# Use GPU acceleration (if available)
vmtranscriber transcribe audio_file.mp3 --device cuda

# Translate non-English speech to English
vmtranscriber transcribe audio_file.mp3 --translate

# Force a specific language
vmtranscriber transcribe audio_file.mp3 --language es

# Skip audio conversion (use original file format)
vmtranscriber transcribe audio_file.mp3 --no-convert

# Custom output formats
vmtranscriber transcribe audio_file.mp3 --formats txt json csv

# Custom output directory
vmtranscriber transcribe audio_file.mp3 --output-dir /path/to/output
```

### Batch Processing

```bash
# Process all audio files in a directory
vmtranscriber batch /path/to/audio/directory

# Process with custom options
vmtranscriber batch /path/to/audio/directory --model medium --recursive

# Custom file patterns
vmtranscriber batch /path/to/audio/directory --pattern "*.mp3,*.m4a"

# Output to specific directory
vmtranscriber batch /path/to/audio/directory --output-dir /path/to/output
```

### Command Line Arguments

| Command | Description |
|---------|-------------|
| `transcribe` | Transcribe a single audio file |
| `batch` | Process multiple audio files in batch |
| `info` | Display information about an audio file |
| `config` | View current configuration |
| `reset-config` | Reset configuration to defaults |

### Transcribe Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `audio_file` | Path to audio file (required) | - |
| `--model, -m` | Model size: tiny, base, small, medium, large-v3 | from config |
| `--device, -d` | Inference device: cpu, cuda | from config |
| `--translate, -t` | Translate non-English to English | from config |
| `--language, -l` | Force specific language code (e.g., en, es) | auto-detect |
| `--no-convert` | Skip FFmpeg WAV conversion | False |
| `--output-dir, -o` | Output directory | same as input |
| `--formats, -f` | Output formats (txt, srt, vtt, json, csv, md) | from config |

### Batch Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_path` | Directory or file path (required) | - |
| `--recursive, -r` | Process subdirectories recursively | False |
| `--pattern, -p` | File pattern to match | *.mp3,*.m4a,*.wav,*.flac |

## 🎯 Model Selection

Choose the model size based on your needs:

- **tiny**: Fastest, least accurate (~39MB)
- **base**: Fast, good for basic transcription (~74MB)
- **small**: Balanced speed/accuracy (~244MB) ⭐ **Recommended**
- **medium**: Better accuracy, slower (~769MB)
- **large-v3**: Best accuracy, slowest (~1550MB)

## 🔧 Configuration

VMTranscriber automatically creates and manages a configuration file with your preferences.

### Configuration File Location
- **Windows**: `%APPDATA%\Local\VMTranscriber\config.yaml`
- **macOS/Linux**: `~/.config/vmtranscriber/config.yaml`

### Key Configuration Options
- **Model settings**: Default model size and device
- **Audio quality**: Sample rate, channels, conversion preferences
- **Output formats**: Which formats to generate by default
- **Transcription options**: VAD filter, language detection, translation

### Performance Tuning

- **CPU Users**: The default `int8` compute type provides good performance on CPU
- **GPU Users**: Use `--device cuda` for significant speed improvements
- **Memory Constrained**: Use smaller models like `tiny` or `base`

### Audio Quality

- **Sample Rate**: Automatically converts to 16kHz for optimal Whisper performance
- **Channels**: Converts to mono for consistent results
- **Format**: WAV conversion ensures maximum compatibility

## 📁 Output Files

VMTranscriber generates multiple output formats to suit different needs:

### Plain Text (.txt)
Simple text transcript without timestamps, perfect for reading or further processing.

### SRT Subtitles (.srt)
Standard subtitle format with timestamps, ideal for video players and editing software.

### VTT Captions (.vtt)
Web Video Text Tracks format, perfect for web applications and HTML5 video.

### JSON (.json)
Structured data with metadata, timestamps, and language information. Perfect for programmatic processing.

### CSV (.csv)
Spreadsheet format with columns for start time, end time, duration, and text. Great for analysis.

### Markdown (.md)
Formatted transcript with timestamps and metadata, perfect for documentation.

## 🌍 Language Support

Whisper supports 99+ languages including:
- English, Spanish, French, German, Italian
- Chinese, Japanese, Korean, Arabic
- Russian, Portuguese, Dutch, Swedish
- And many more!

## 🐛 Troubleshooting

### Common Issues

1. **"faster-whisper is not installed"**
   ```bash
   pip install faster-whisper
   ```

2. **"No module named 'click'"**
   ```bash
   pip install click tqdm PyYAML
   ```

2. **Audio conversion fails**
   - Ensure FFmpeg is installed and in your PATH
   - Use `--no-convert` to skip conversion

3. **Out of memory errors**
   - Use a smaller model (tiny, base, small)
   - Close other applications to free memory

4. **Slow performance on CPU**
   - Consider using a smaller model
   - Ensure you have sufficient RAM

### Performance Tips

- **Windows Users**: Install PyTorch CPU version for better performance:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```
- **GPU Users**: Ensure CUDA is properly installed and PyTorch supports it

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - The amazing speech recognition model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper implementation
- [FFmpeg](https://ffmpeg.org/) - Audio/video processing toolkit
- [Click](https://click.palletsprojects.com/) - Beautiful command line interface creation kit
- [tqdm](https://tqdm.github.io/) - Fast, extensible progress bar

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information about your problem

---

**Happy Transcribing! 🎵✨**
