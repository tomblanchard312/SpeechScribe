"""
Core transcription functionality for VMTranscriber.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import time

from .audio import AudioProcessor
from .output import OutputFormatter

logger = logging.getLogger(__name__)

class TranscriptionError(Exception):
    """Custom exception for transcription errors."""
    pass

class TranscriptionManager:
    """Manages the transcription process with progress tracking and error handling."""
    
    def __init__(self, config):
        """Initialize transcription manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.audio_processor = AudioProcessor(config)
        self.output_formatter = OutputFormatter(config)
        self.model = None
        
    def _load_model(self, model_name: str, device: str):
        """Load the Whisper model.
        
        Args:
            model_name: Name of the model to load
            device: Device to use (cpu/cuda)
            
        Raises:
            TranscriptionError: If model loading fails
        """
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"Loading Whisper model: {model_name} on {device}")
            
            compute_type = "int8" if device == "cpu" else "float16"
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
            
            logger.info(f"Model loaded successfully: {model_name}")
            
        except ImportError as e:
            raise TranscriptionError(
                "faster-whisper is not installed.\n"
                "Install with: pip install faster-whisper\n\n"
                "On Windows CPU only, you can accelerate installs like:\n"
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install faster-whisper"
            ) from e
        except Exception as e:
            raise TranscriptionError(f"Failed to load model {model_name}: {e}") from e
    
    def transcribe_audio(self, audio_path: Path, **kwargs) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Transcribe a single audio file.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional transcription options
            
        Returns:
            Tuple of (segments, metadata)
            
        Raises:
            TranscriptionError: If transcription fails
        """
        try:
            # Validate and prepare audio
            logger.info(f"Starting transcription of: {audio_path.name}")
            
            # Get model configuration
            model_config = self.config.get_model_config()
            model_name = kwargs.get('model', model_config.get('model', 'small'))
            device = kwargs.get('device', model_config.get('device', 'cpu'))
            translate = kwargs.get('translate', model_config.get('translate', False))
            language = kwargs.get('language', model_config.get('language'))
            no_convert = kwargs.get('no_convert', False)
            
            # Load model if not already loaded or if model changed
            if (self.model is None or 
                getattr(self.model, 'model_size', None) != model_name):
                self._load_model(model_name, device)
            
            # Prepare audio file
            if no_convert:
                prepared_audio = audio_path
                logger.info("Skipping audio conversion as requested")
            else:
                try:
                    prepared_audio = self.audio_processor.prepare_audio(audio_path)
                except Exception as e:
                    logger.warning(f"Audio conversion failed: {e}. Using original file.")
                    prepared_audio = audio_path
            
            # Get audio duration for progress bar
            audio_info = self.audio_processor.get_audio_info(prepared_audio)
            duration = audio_info.get('duration', 0) if audio_info else 0
            
            # Start transcription with progress bar
            logger.info("Starting transcription...")
            
            task = "translate" if translate else "transcribe"
            vad_filter = model_config.get('vad_filter', True)
            min_silence_duration_ms = model_config.get('min_silence_duration_ms', 300)
            
            segments_iter, info = self.model.transcribe(
                str(prepared_audio),
                task=task,
                language=language,
                vad_filter=vad_filter,
                vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms)
            )
            
            # Process segments with progress bar
            segments = []
            with tqdm(
                desc="Transcribing",
                unit="seg",
                total=None,  # Unknown total segments
                dynamic_ncols=True
            ) as pbar:
                for segment in segments_iter:
                    segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })
                    pbar.update(1)
                    pbar.set_postfix({
                        "time": f"{segment.end:.1f}s",
                        "segments": len(segments)
                    })
            
            # Prepare metadata
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": duration,
                "segments_count": len(segments),
                "model": model_name,
                "device": device,
                "task": task
            }
            
            logger.info(f"Transcription completed: {len(segments)} segments")
            if info.language:
                logger.info(f"Detected language: {info.language} (prob {info.language_probability:.2f})")
            
            return segments, metadata
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise TranscriptionError(f"Failed to transcribe {audio_path}: {e}") from e
    
    def process_single_file(self, audio_path: Path, output_dir: Optional[Path] = None, **kwargs) -> List[Path]:
        """Process a single audio file and generate outputs.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Output directory. If None, uses input file directory
            **kwargs: Additional transcription options
            
        Returns:
            List of created output file paths
        """
        try:
            # Transcribe audio
            segments, metadata = self.transcribe_audio(audio_path, **kwargs)
            
            # Determine output paths
            if output_dir is None:
                output_dir = audio_path.parent
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            output_config = self.config.get_output_config()
            filename_template = output_config.get('filename_template', '{input_name}_transcript')
            input_name = audio_path.stem
            output_base = output_dir / filename_template.format(input_name=input_name)
            
            # Write outputs
            output_files = self.output_formatter.write_all_formats(
                segments, output_base, metadata=metadata
            )
            
            logger.info(f"Generated {len(output_files)} output files for {audio_path.name}")
            return output_files
            
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            raise
    
    def batch_transcribe(self, audio_paths: List[Path], output_dir: Optional[Path] = None, 
                        **kwargs) -> Dict[Path, List[Path]]:
        """Process multiple audio files in batch.
        
        Args:
            audio_paths: List of audio file paths
            output_dir: Output directory for all files
            **kwargs: Additional transcription options
            
        Returns:
            Dictionary mapping input files to their output files
        """
        if not audio_paths:
            logger.warning("No audio files provided for batch processing")
            return {}
        
        logger.info(f"Starting batch transcription of {len(audio_paths)} files")
        
        results = {}
        failed_files = []
        
        # Process files with overall progress bar
        with tqdm(
            total=len(audio_paths),
            desc="Processing files",
            unit="file",
            dynamic_ncols=True
        ) as pbar:
            for audio_path in audio_paths:
                try:
                    pbar.set_description(f"Processing {audio_path.name}")
                    
                    # Process single file
                    output_files = self.process_single_file(audio_path, output_dir, **kwargs)
                    results[audio_path] = output_files
                    
                    pbar.set_postfix({
                        "success": len(results),
                        "failed": len(failed_files)
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process {audio_path}: {e}")
                    failed_files.append((audio_path, str(e)))
                    results[audio_path] = []
                
                pbar.update(1)
        
        # Summary
        successful = len([f for f, outputs in results.items() if outputs])
        logger.info(f"Batch processing completed: {successful}/{len(audio_paths)} files successful")
        
        if failed_files:
            logger.warning(f"Failed files: {len(failed_files)}")
            for file_path, error in failed_files:
                logger.warning(f"  {file_path.name}: {error}")
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if self.model:
            del self.model
            self.model = None
        logger.debug("Transcription manager cleaned up")

# Convenience functions
def transcribe_audio(audio_path: str, **kwargs) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convenience function to transcribe a single audio file.
    
    Args:
        audio_path: Path to audio file
        **kwargs: Additional options
        
    Returns:
        Tuple of (segments, metadata)
    """
    from .config import Config
    
    config = Config()
    manager = TranscriptionManager(config)
    
    try:
        return manager.transcribe_audio(Path(audio_path), **kwargs)
    finally:
        manager.cleanup()

def batch_transcribe(audio_paths: List[str], **kwargs) -> Dict[Path, List[Path]]:
    """Convenience function for batch transcription.
    
    Args:
        audio_paths: List of audio file paths
        **kwargs: Additional options
        
    Returns:
        Dictionary mapping input files to output files
    """
    from .config import Config
    
    config = Config()
    manager = TranscriptionManager(config)
    
    try:
        return manager.batch_transcribe([Path(p) for p in audio_paths], **kwargs)
    finally:
        manager.cleanup()
