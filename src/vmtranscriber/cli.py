"""
Command Line Interface for VMTranscriber.
"""

import click
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config
from .core import TranscriptionManager, TranscriptionError
from .audio import AudioProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vmtranscriber.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool, quiet: bool):
    """Set up logging based on verbosity flags."""
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

@click.group()
@click.version_option(version="1.0.0", prog_name="VMTranscriber")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Suppress all output except errors')
@click.pass_context
def cli(ctx, verbose, quiet):
    """VMTranscriber - Offline Voice Mail Transcription Tool
    
    A powerful, offline-capable audio transcription tool built with OpenAI's Whisper technology.
    """
    setup_logging(verbose, quiet)
    ctx.ensure_object(dict)
    ctx.obj['config'] = Config()

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True, path_type=Path))
@click.option('--model', '-m', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large-v3']),
              help='Whisper model size (default: from config)')
@click.option('--device', '-d',
              type=click.Choice(['cpu', 'cuda']),
              help='Inference device (default: from config)')
@click.option('--translate', '-t', is_flag=True,
              help='Translate non-English speech to English')
@click.option('--language', '-l', help='Force specific language code (e.g., en, es)')
@click.option('--no-convert', is_flag=True,
              help='Skip audio conversion to WAV')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory (default: same as input file)')
@click.option('--formats', '-f',
              type=click.Choice(['txt', 'srt', 'vtt', 'json', 'csv', 'md']),
              multiple=True,
              help='Output formats (default: from config)')
@click.pass_context
def transcribe(ctx, audio_file, model, device, translate, language, no_convert, 
               output_dir, formats):
    """Transcribe a single audio file."""
    config = ctx.obj['config']
    
    try:
        # Override config with CLI options
        if model:
            config.set('model', model)
        if device:
            config.set('device', device)
        if translate:
            config.set('transcription.translate', True)
        if language:
            config.set('transcription.language', language)
        if formats:
            config.set('output_formats', list(formats))
        
        # Create transcription manager
        manager = TranscriptionManager(config)
        
        try:
            # Process the file
            output_files = manager.process_single_file(
                audio_file,
                output_dir=output_dir,
                no_convert=no_convert
            )
            
            # Success message
            click.echo(f"\n✅ Transcription completed successfully!")
            click.echo(f"📁 Generated {len(output_files)} output files:")
            for output_file in output_files:
                click.echo(f"   • {output_file}")
                
        finally:
            manager.cleanup()
            
    except TranscriptionError as e:
        click.echo(f"❌ Transcription failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        logger.exception("Unexpected error during transcription")
        sys.exit(1)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory for all files')
@click.option('--model', '-m',
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large-v3']),
              help='Whisper model size')
@click.option('--device', '-d',
              type=click.Choice(['cpu', 'cuda']),
              help='Inference device')
@click.option('--translate', '-t', is_flag=True,
              help='Translate non-English speech to English')
@click.option('--language', '-l', help='Force specific language code')
@click.option('--no-convert', is_flag=True,
              help='Skip audio conversion to WAV')
@click.option('--formats', '-f',
              type=click.Choice(['txt', 'srt', 'vtt', 'json', 'csv', 'md']),
              multiple=True,
              help='Output formats')
@click.option('--recursive', '-r', is_flag=True,
              help='Process subdirectories recursively')
@click.option('--pattern', '-p', default='*.mp3,*.m4a,*.wav,*.flac',
              help='File pattern to match (comma-separated)')
@click.pass_context
def batch(ctx, input_path, output_dir, model, device, translate, language,
          no_convert, formats, recursive, pattern):
    """Process multiple audio files in batch."""
    config = ctx.obj['config']
    
    try:
        # Override config with CLI options
        if model:
            config.set('model', model)
        if device:
            config.set('device', device)
        if translate:
            config.set('transcription.translate', True)
        if language:
            config.set('transcription.language', language)
        if formats:
            config.set('output_formats', list(formats))
        
        # Find audio files
        audio_files = find_audio_files(input_path, recursive, pattern)
        
        if not audio_files:
            click.echo("❌ No audio files found matching the criteria.", err=True)
            sys.exit(1)
        
        click.echo(f"🎵 Found {len(audio_files)} audio files to process")
        
        # Create transcription manager
        manager = TranscriptionManager(config)
        
        try:
            # Process files in batch
            results = manager.batch_transcribe(
                audio_files,
                output_dir=output_dir,
                no_convert=no_convert
            )
            
            # Summary
            successful = len([f for f, outputs in results.items() if outputs])
            failed = len(audio_files) - successful
            
            click.echo(f"\n📊 Batch processing completed!")
            click.echo(f"✅ Successful: {successful}")
            if failed > 0:
                click.echo(f"❌ Failed: {failed}")
            
            if output_dir:
                click.echo(f"📁 Output directory: {output_dir}")
                
        finally:
            manager.cleanup()
            
    except Exception as e:
        click.echo(f"❌ Batch processing failed: {e}", err=True)
        logger.exception("Unexpected error during batch processing")
        sys.exit(1)

@cli.command()
@click.option('--config-path', type=click.Path(path_type=Path),
              help='Path to configuration file')
@click.pass_context
def config(ctx, config_path):
    """Manage configuration settings."""
    config = ctx.obj['config']
    
    click.echo(f"📁 Configuration file: {config.config_path}")
    click.echo(f"🔧 Current settings:")
    
    # Display current configuration
    for key, value in config.config.items():
        if isinstance(value, dict):
            click.echo(f"  {key}:")
            for subkey, subvalue in value.items():
                click.echo(f"    {subkey}: {subvalue}")
        else:
            click.echo(f"  {key}: {value}")

@cli.command()
@click.option('--config-path', type=click.Path(path_type=Path),
              help='Path to configuration file')
@click.pass_context
def reset_config(ctx, config_path):
    """Reset configuration to default values."""
    config = ctx.obj['config']
    
    if click.confirm("Are you sure you want to reset all configuration to defaults?"):
        config.reset_to_defaults()
        click.echo("✅ Configuration reset to defaults")

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True, path_type=Path))
@click.pass_context
def info(ctx, audio_file):
    """Display information about an audio file."""
    config = ctx.obj['config']
    audio_processor = AudioProcessor(config)
    
    try:
        # Validate file
        is_valid, error_msg = audio_processor.validate_audio_file(audio_file)
        if not is_valid:
            click.echo(f"❌ Invalid audio file: {error_msg}", err=True)
            sys.exit(1)
        
        # Get audio info
        audio_info = audio_processor.get_audio_info(audio_file)
        
        click.echo(f"📁 File: {audio_file}")
        click.echo(f"📏 Size: {audio_file.stat().st_size / 1024:.1f} KB")
        click.echo(f"🔊 Format: {audio_file.suffix.upper()}")
        
        if audio_info:
            click.echo(f"⏱️  Duration: {audio_info['duration']:.2f} seconds")
            click.echo(f"🎵 Sample Rate: {audio_info['sample_rate']} Hz")
            click.echo(f"🔊 Channels: {audio_info['channels']}")
            click.echo(f"💾 Codec: {audio_info['codec']}")
            if audio_info.get('bit_rate'):
                click.echo(f"📊 Bit Rate: {audio_info['bit_rate']} bps")
        else:
            click.echo("ℹ️  Audio info not available (FFmpeg required)")
            
        # Check FFmpeg availability
        if audio_processor.ffmpeg_available:
            click.echo("✅ FFmpeg available for audio conversion")
        else:
            click.echo("⚠️  FFmpeg not available - audio conversion limited")
            
    except Exception as e:
        click.echo(f"❌ Error getting file info: {e}", err=True)
        sys.exit(1)

def find_audio_files(input_path: Path, recursive: bool, pattern: str) -> List[Path]:
    """Find audio files matching the pattern."""
    patterns = [p.strip() for p in pattern.split(',')]
    audio_files = []
    
    if input_path.is_file():
        # Single file
        if any(input_path.match(p) for p in patterns):
            audio_files.append(input_path)
    elif input_path.is_dir():
        # Directory
        if recursive:
            glob_pattern = "**/*"
        else:
            glob_pattern = "*"
        
        for pattern in patterns:
            audio_files.extend(input_path.glob(glob_pattern))
    
    # Filter to only include files that exist and match patterns
    audio_files = [f for f in audio_files if f.is_file() and any(f.match(p) for p in patterns)]
    
    # Remove duplicates and sort
    audio_files = sorted(set(audio_files))
    
    return audio_files

if __name__ == '__main__':
    cli()
