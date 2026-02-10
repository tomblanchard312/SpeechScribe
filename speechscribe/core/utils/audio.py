"""
Audio processing utilities for VMTranscriber.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple
import wave
import contextlib

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio file processing and conversion."""
    
    SUPPORTED_FORMATS = {'.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.wma', '.mov', '.mp4', '.avi', '.mkv', '.webm', '.m4v'}
    
    def __init__(self, config):
        """Initialize audio processor with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.ffmpeg_available = self._check_ffmpeg()
        
        if not self.ffmpeg_available:
            logger.warning("FFmpeg not found. Audio conversion will be limited.")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available in the system PATH."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL, 
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def validate_audio_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate if the audio file can be processed.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"
        
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported audio format: {suffix}. Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, f"File is empty: {file_path}"
        
        if file_size < 1024:  # Less than 1KB
            return False, f"File is too small to be valid audio: {file_path}"
        
        return True, ""
    
    def get_audio_info(self, file_path: Path) -> Optional[dict]:
        """Get basic information about an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with audio information or None if failed
        """
        if not self.ffmpeg_available:
            logger.warning("Cannot get audio info without FFmpeg")
            return None
        
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            
            # Extract relevant information
            audio_stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'audio'), None)
            
            if audio_stream:
                return {
                    'duration': float(info.get('format', {}).get('duration', 0)),
                    'sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'channels': int(audio_stream.get('channels', 0)),
                    'codec': audio_stream.get('codec_name', 'unknown'),
                    'bit_rate': int(audio_stream.get('bit_rate', 0))
                }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to get audio info for {file_path}: {e}")
        
        return None
    
    def convert_to_wav(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        """Convert audio file to WAV format optimized for Whisper.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file. If None, creates in same directory
            
        Returns:
            Path to the converted WAV file
            
        Raises:
            RuntimeError: If conversion fails
        """
        if not self.ffmpeg_available:
            raise RuntimeError(
                "FFmpeg is required for audio conversion. "
                "Please install FFmpeg or use --no-convert flag."
            )
        
        if output_path is None:
            output_path = input_path.with_suffix('.wav')
        
        # Get audio configuration
        audio_config = self.config.get_audio_config()
        sample_rate = audio_config.get('sample_rate', 16000)
        channels = audio_config.get('channels', 1)
        
        try:
            cmd = [
                "ffmpeg", "-y",  # Overwrite output file
                "-i", str(input_path),
                "-ac", str(channels),  # Audio channels
                "-ar", str(sample_rate),  # Sample rate
                "-acodec", "pcm_s16le",  # 16-bit PCM
                str(output_path)
            ]
            
            logger.info(f"Converting {input_path.name} to WAV format...")
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Successfully converted to: {output_path}")
                return output_path
            else:
                raise RuntimeError("Conversion completed but output file is invalid")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg conversion failed: {e}"
            if e.stderr:
                error_msg += f"\nFFmpeg error: {e.stderr}"
            raise RuntimeError(error_msg)
        except Exception as e:
            raise RuntimeError(f"Unexpected error during conversion: {e}")
    
    def prepare_audio(self, input_path: Path, force_convert: bool = False) -> Path:
        """Prepare audio file for transcription.
        
        Args:
            input_path: Path to input audio file
            force_convert: Force conversion even if already WAV
            
        Returns:
            Path to prepared audio file
        """
        # Validate input file
        is_valid, error_msg = self.validate_audio_file(input_path)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Check if conversion is needed
        if input_path.suffix.lower() == '.wav' and not force_convert:
            logger.info(f"Using existing WAV file: {input_path}")
            return input_path
        
        # Convert to WAV
        try:
            return self.convert_to_wav(input_path)
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            if not self.ffmpeg_available:
                logger.warning("FFmpeg not available. Trying to use original file...")
                return input_path
            else:
                raise
    
    def cleanup_temp_files(self, temp_files: list):
        """Clean up temporary audio files.
        
        Args:
            temp_files: List of temporary file paths to remove
        """
        for temp_file in temp_files:
            try:
                if temp_file.exists() and temp_file != temp_file.parent / temp_file.name:
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
