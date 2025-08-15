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
from .voice_synthesis import VoiceSynthesizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('speechscribe.log', encoding='utf-8')
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
@click.version_option(version="1.0.0", prog_name="SpeechScribe")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Suppress all output except errors')
@click.pass_context
def cli(ctx, verbose, quiet):
    """SpeechScribe - Comprehensive Speech Processing Tool
    
    A powerful, offline-capable speech processing tool built with OpenAI's Whisper technology and advanced TTS capabilities.
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
@click.option('--pattern', '-p', default='*.mp3,*.m4a,*.wav,*.flac,*.mov,*.mp4',
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

@cli.command()
@click.argument('text', type=str)
@click.option('--output', '-o', type=click.Path(path_type=Path), required=True,
               help='Output audio file path')
@click.option('--voice', '-v', default='default', help='Voice name to use')
@click.option('--engine', '-e', default='coqui_tts',
               type=click.Choice(['coqui_tts', 'elevenlabs', 'azure_speech']),
               help='TTS engine to use')
@click.option('--api-key', help='API key for cloud services')
@click.option('--no-fix-speed', is_flag=True, help='Skip audio speed correction')
@click.option('--quality', '-q', type=click.Choice(['low', 'medium', 'high']), default='high',
               help='Voice quality level (affects inflection and naturalness)')
@click.option('--emotion', type=click.Choice(['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']),
               help='Emotional tone for more natural inflection')
@click.option('--speed', type=click.FloatRange(0.5, 2.0), default=1.0,
               help='Speech rate (0.5 = slow, 2.0 = fast)')
@click.option('--pitch', type=click.IntRange(-12, 12), default=0,
               help='Pitch adjustment in semitones (-12 to +12)')
@click.option('--emphasis', type=click.Choice(['weak', 'normal', 'strong']), default='normal',
               help='Word emphasis level for better inflection')
@click.pass_context
def speak(ctx, text, output, voice, engine, api_key, no_fix_speed, quality, emotion, speed, pitch, emphasis):
    """Convert text to speech."""
    config = ctx.obj['config']
    
    try:
        synthesizer = VoiceSynthesizer(config)
        
        # Check if engine is available
        if not synthesizer.available_engines.get(engine, False):
            click.echo(f"❌ Engine {engine} is not available.", err=True)
            click.echo("Available engines:")
            for eng, available in synthesizer.available_engines.items():
                status = "✅" if available else "❌"
                click.echo(f"  {status} {eng}")
            sys.exit(1)
        
        # Check if this is a trained voice
        if voice != "default":
            trained_voices = synthesizer.list_trained_voices()
            voice_names = [v.get('name') for v in trained_voices]
            
            if voice in voice_names:
                click.echo(f"🎭 Using trained voice model: {voice}")
                output_path = synthesizer.use_trained_voice(voice, text, output, fix_speed=not no_fix_speed)
                click.echo(f"✅ Speech generated successfully using trained voice!")
                click.echo(f"📁 Output file: {output_path}")
                return
            else:
                click.echo(f"⚠️  Voice '{voice}' not found in trained voices, using default TTS")
        
        # Generate speech with regular TTS
        click.echo(f"🎤 Converting text to speech using {engine}...")
        
        kwargs = {}
        if api_key:
            if engine == 'elevenlabs':
                kwargs['api_key'] = api_key
            elif engine == 'azure_speech':
                # For Azure, you might want to split the API key
                kwargs['subscription_key'] = api_key
        
        # Add speed fixing option
        kwargs['fix_speed'] = not no_fix_speed
        
        # Add enhanced voice quality options
        if quality:
            synthesizer.set_voice_quality(quality)
        
        # Add enhanced parameters for better naturalness
        if emotion:
            kwargs['emotion'] = emotion
        if speed != 1.0:
            kwargs['speed'] = speed
        if pitch != 0:
            kwargs['pitch'] = pitch
        if emphasis != 'normal':
            kwargs['emphasis'] = emphasis
        
        # Add quality setting
        kwargs['quality'] = quality
        
        output_path = synthesizer.text_to_speech(text, output, voice, engine, **kwargs)
        
        click.echo(f"✅ Speech generated successfully!")
        click.echo(f"📁 Output file: {output_path}")
        
    except Exception as e:
        click.echo(f"❌ Speech synthesis failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('source_audio', type=click.Path(exists=True, path_type=Path))
@click.argument('text', type=str)
@click.option('--output', '-o', type=click.Path(path_type=Path), required=True,
              help='Output audio file path')
@click.option('--engine', '-e', default='coqui_tts',
              type=click.Choice(['coqui_tts']),
              help='Voice cloning engine to use')
@click.option('--no-fix-speed', is_flag=True, help='Skip audio speed correction')
@click.option('--quality', '-q', type=click.Choice(['low', 'medium', 'high']), default='high',
              help='Voice quality level (affects inflection and naturalness)')
@click.option('--emotion', type=click.Choice(['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']),
              help='Emotional tone for more natural inflection')
@click.pass_context
def clone_voice(ctx, source_audio, text, output, engine, no_fix_speed, quality, emotion):
    """Clone a voice from source audio and use it for new text."""
    config = ctx.obj['config']
    
    try:
        synthesizer = VoiceSynthesizer(config)
        
        # Check if engine is available
        if not synthesizer.available_engines.get(engine, False):
            click.echo(f"❌ Voice cloning engine {engine} is not available.", err=True)
            sys.exit(1)
        
        # Clone voice
        click.echo(f"🎭 Cloning voice from {source_audio.name}...")
        click.echo(f"📝 Generating new speech: {text}")
        
        # Add speed fixing option
        kwargs = {'fix_speed': not no_fix_speed}
        
        # Add quality and emotion options
        if quality:
            synthesizer.set_voice_quality(quality)
        if emotion:
            kwargs['emotion'] = emotion
        
        output_path = synthesizer.clone_voice(source_audio, text, output, engine, **kwargs)
        
        click.echo(f"✅ Voice cloned and speech generated successfully!")
        click.echo(f"📁 Output file: {output_path}")
        
    except Exception as e:
        click.echo(f"❌ Voice cloning failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('source_audio', type=click.Path(exists=True, path_type=Path))
@click.argument('target_voice', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), required=True,
              help='Output audio file path')
@click.option('--engine', '-e', default='so_vits_svc',
              type=click.Choice(['so_vits_svc', 'rvc']),
              help='Voice conversion engine to use')
@click.option('--no-fix-speed', is_flag=True, help='Skip audio speed correction')
@click.pass_context
def convert_voice(ctx, source_audio, target_voice, output, engine, no_fix_speed):
    """Convert one voice to another voice."""
    config = ctx.obj['config']
    
    try:
        synthesizer = VoiceSynthesizer(config)
        
        # Check if engine is available
        if not synthesizer.available_engines.get(engine, False):
            click.echo(f"❌ Voice conversion engine {engine} is not available.", err=True)
            sys.exit(1)
        
        # Convert voice
        click.echo(f"🔄 Converting voice from {source_audio.name} to {target_voice.name}...")
        
        # Add speed fixing option
        kwargs = {'fix_speed': not no_fix_speed}
        output_path = synthesizer.convert_voice(source_audio, target_voice, output, engine, **kwargs)
        
        click.echo(f"✅ Voice conversion completed successfully!")
        click.echo(f"📁 Output file: {output_path}")
        
    except Exception as e:
        click.echo(f"❌ Voice conversion failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--engine', '-e', default='coqui_tts',
              type=click.Choice(['coqui_tts', 'elevenlabs', 'azure_speech']),
              help='TTS engine to query')
@click.pass_context
def voices(ctx, engine):
    """List available voices for a specific engine."""
    config = ctx.obj['config']
    
    try:
        synthesizer = VoiceSynthesizer(config)
        
        click.echo(f"🎵 Available voices for {engine}:")
        
        voices_list = synthesizer.get_available_voices(engine)
        if voices_list:
            for voice in voices_list[:20]:  # Limit to first 20
                click.echo(f"  • {voice}")
            if len(voices_list) > 20:
                click.echo(f"  ... and {len(voices_list) - 20} more")
        else:
            click.echo("  No voices available")
            
    except Exception as e:
        click.echo(f"❌ Failed to get voices: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def voice_engines(ctx):
    """Show information about available voice synthesis engines."""
    config = ctx.obj['config']
    
    try:
        synthesizer = VoiceSynthesizer(config)
        engine_info = synthesizer.get_engine_info()
        
        click.echo("🔊 Voice Synthesis Engines:")
        click.echo("=" * 50)
        
        for engine_name, info in engine_info.items():
            if engine_name == 'available_engines':
                continue
                
            status = "✅" if info['available'] else "❌"
            capabilities = ", ".join(info['capabilities'])
            offline = "🟢 Offline" if info['offline'] else "🔴 Online"
            quality = info['quality'].replace('_', ' ').title()
            
            click.echo(f"\n{status} {engine_name.upper()}")
            click.echo(f"  Capabilities: {capabilities}")
            click.echo(f"  Mode: {offline}")
            click.echo(f"  Quality: {quality}")
            
            if info.get('requires_api_key'):
                click.echo(f"  ⚠️  Requires API key")
            elif info.get('requires_credentials'):
                click.echo(f"  ⚠️  Requires Azure credentials")
        
    except Exception as e:
        click.echo(f"❌ Failed to get engine info: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('input_folder', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), required=True,
              help='Output voice model directory')
@click.option('--engine', '-e', default='coqui_tts',
              type=click.Choice(['coqui_tts']),
              help='Voice training engine to use')
@click.option('--quality', '-q', type=click.Choice(['low', 'medium', 'high', 'ultra']), 
               default='high', help='Voice quality level (ultra for maximum realism)')
@click.option('--min-duration', default=5.0, help='Minimum audio duration in seconds (default: 5.0)')
@click.option('--max-duration', default=300.0, help='Maximum audio duration in seconds (default: 300.0 = 5 minutes)')
@click.option('--recursive', '-r', is_flag=True, help='Process subdirectories recursively')
@click.option('--pattern', '-p', default='*.mp3,*.m4a,*.wav,*.flac,*.mp4',
              help='Audio file pattern to match')
@click.pass_context
def train_voice(ctx, input_folder, output, engine, quality, min_duration, max_duration, recursive, pattern):
    """Train a reusable voice model from multiple audio files."""
    config = ctx.obj['config']
    
    try:
        synthesizer = VoiceSynthesizer(config)
        
        # Check if engine is available
        if not synthesizer.available_engines.get(engine, False):
            click.echo(f"❌ Voice training engine {engine} is not available.", err=True)
            sys.exit(1)
        
        # Find audio files
        audio_files = find_audio_files(input_folder, recursive, pattern)
        
        if not audio_files:
            click.echo("❌ No audio files found matching the criteria.", err=True)
            sys.exit(1)
        
        click.echo(f"🎵 Found {len(audio_files)} audio files for voice training")
        click.echo(f"📏 Duration filter: {min_duration:.1f}s - {max_duration:.1f}s")
        
        # Filter files by duration and quality
        valid_files = []
        for file_path in audio_files:
            try:
                # Get audio info
                audio_processor = AudioProcessor(config)
                audio_info = audio_processor.get_audio_info(file_path)
                
                if audio_info and min_duration <= audio_info['duration'] <= max_duration:
                    valid_files.append((file_path, audio_info['duration']))
                    click.echo(f"✅ Accepting {file_path.name} (duration: {audio_info['duration']:.1f}s)")
                else:
                    if audio_info:
                        if audio_info['duration'] < min_duration:
                            click.echo(f"⚠️  Skipping {file_path.name} (too short: {audio_info['duration']:.1f}s < {min_duration:.1f}s)")
                        else:
                            click.echo(f"⚠️  Skipping {file_path.name} (too long: {audio_info['duration']:.1f}s > {max_duration:.1f}s)")
                    else:
                        click.echo(f"⚠️  Skipping {file_path.name} (could not get duration)")
            except Exception as e:
                click.echo(f"⚠️  Skipping {file_path.name} (error: {e})")
        
        if not valid_files:
            click.echo("ERROR: No valid audio files found after filtering", err=True)
            click.echo(f"TIP: Try adjusting duration limits: --min-duration {min_duration/2:.1f} --max-duration {max_duration*2:.1f}")
            sys.exit(1)
        
        click.echo(f"✅ {len(valid_files)} files passed quality checks")
        
        # Sort by duration (longer files first for better training)
        valid_files.sort(key=lambda x: x[1], reverse=True)
        
        # Train the voice model
        click.echo(f"🎭 Training voice model from {len(valid_files)} audio files...")
        click.echo(f"📁 Output directory: {output}")
        
        # Set quality preferences
        if quality:
            synthesizer.set_voice_quality(quality)
        
        # Train the model
        voice_model_path = synthesizer.train_voice_model(
            audio_files=[f[0] for f in valid_files],
            output_dir=output,
            engine=engine,
            quality=quality
        )
        
        click.echo(f"✅ Voice model trained successfully!")
        click.echo(f"📁 Model saved to: {voice_model_path}")
        click.echo(f"🎤 You can now use this voice with: speechscribe speak 'text' --output file.wav --voice {output.name}")
        click.echo(f"🧪 Test voice quality with: speechscribe test-voice {output.name}")
        click.echo(f"🚀 For better naturalness, try: speechscribe train-voice-enhanced '{input_folder}' '{output.name}_enhanced' --quality {quality}")
        
    except Exception as e:
        click.echo(f"❌ Voice training failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output', type=click.Path(path_type=Path))
@click.option('--engine', '-e', default='coqui_tts', 
              type=click.Choice(['coqui_tts']), help='Training engine to use')
@click.option('--quality', '-q', default='high', 
              type=click.Choice(['low', 'medium', 'high', 'ultra']), help='Quality level for maximum naturalness')
@click.option('--min-duration', default=2.0, help='Minimum audio duration in seconds (default: 2.0 for optimal speech)')
@click.option('--max-duration', default=60.0, help='Maximum audio duration in seconds (default: 60.0 for natural speech)')
@click.option('--recursive', '-r', is_flag=True, help='Process subdirectories recursively')
@click.option('--pattern', '-p', default='*.mp3,*.m4a,*.wav,*.flac,*.mp4',
              help='Audio file pattern to match')
@click.pass_context
def train_voice_enhanced(ctx, input_folder, output, engine, quality, min_duration, max_duration, recursive, pattern):
    """Train a voice model with enhanced focus on naturalness, pitch variation, and timing."""
    config = ctx.obj['config']
    
    try:
        synthesizer = VoiceSynthesizer(config)
        
        # Check if engine is available
        if not synthesizer.available_engines.get(engine, False):
            click.echo(f"❌ Voice training engine {engine} is not available.", err=True)
            sys.exit(1)
        
        # Find audio files
        audio_files = find_audio_files(input_folder, recursive, pattern)
        
        if not audio_files:
            click.echo("❌ No audio files found matching the criteria.", err=True)
            sys.exit(1)
        
        click.echo(f"🎵 Found {len(audio_files)} audio files for enhanced naturalness training")
        click.echo(f"📏 Duration filter: {min_duration:.1f}s - {max_duration:.1f}s (optimized for natural speech)")
        click.echo(f"🌿 Quality focus: Maximum naturalness and expression")
        
        # Filter files by duration and naturalness
        valid_files = []
        for file_path in audio_files:
            try:
                # Get audio info
                audio_processor = AudioProcessor(config)
                audio_info = audio_processor.get_audio_info(file_path)
                
                if audio_info and min_duration <= audio_info['duration'] <= max_duration:
                    valid_files.append((file_path, audio_info['duration']))
                    click.echo(f"✅ Accepting {file_path.name} (duration: {audio_info['duration']:.1f}s)")
                else:
                    if audio_info:
                        if audio_info['duration'] < min_duration:
                            click.echo(f"⚠️  Skipping {file_path.name} (too short: {audio_info['duration']:.1f}s < {min_duration:.1f}s)")
                        else:
                            click.echo(f"⚠️  Skipping {file_path.name} (too long: {audio_info['duration']:.1f}s > {max_duration:.1f}s)")
                    else:
                        click.echo(f"⚠️  Skipping {file_path.name} (could not get duration)")
            except Exception as e:
                click.echo(f"⚠️  Skipping {file_path.name} (error: {e})")
        
        if not valid_files:
            click.echo("ERROR: No valid audio files found after filtering", err=True)
            click.echo(f"TIP: Try adjusting duration limits: --min-duration {min_duration/2:.1f} --max-duration {max_duration*2:.1f}")
            sys.exit(1)
        
        click.echo(f"✅ {len(valid_files)} files passed naturalness checks")
        
        # Sort by duration (optimal speech length first)
        valid_files.sort(key=lambda x: abs(x[1] - 5.0))  # Sort by closeness to optimal 5-second duration
        
        # Train the enhanced voice model
        click.echo(f"🎭 Training enhanced naturalness voice model from {len(valid_files)} audio files...")
        click.echo(f"📁 Output directory: {output}")
        click.echo(f"🌿 Naturalness features: Enhanced pitch variation, breathing patterns, emotional expression")
        
        # Set quality preferences
        if quality:
            synthesizer.set_voice_quality(quality)
        
        # Train the enhanced model
        voice_model_path = synthesizer.train_voice_model_enhanced(
            audio_files=[f[0] for f in valid_files],
            output_dir=output,
            engine=engine,
            quality=quality
        )
        
        click.echo(f"🎉 Enhanced naturalness voice model trained successfully!")
        click.echo(f"📁 Model saved to: {voice_model_path}")
        click.echo(f"🌿 Naturalness score: {voice_model_path.name}")
        click.echo(f"🎤 Test the improved voice with: speechscribe test-voice {output.name}")
        click.echo(f"💡 This model focuses on: Natural pitch variation, timing, and emotional expression")
        
    except Exception as e:
        click.echo(f"❌ Enhanced voice training failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('voice_name', type=str)
@click.option('--text', '-t', type=str, 
              default="Hello, this is a test of the voice quality. I'm checking for naturalness, pitch variation, and timing.",
              help='Test text to synthesize')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output audio file path (default: test_output_{voice_name}.wav)')
@click.pass_context
def test_voice(ctx, voice_name, text, output):
    """Test the quality of a trained voice and provide feedback on naturalness, pitch, and timing."""
    config = ctx.obj['config']
    
    try:
        synthesizer = VoiceSynthesizer(config)
        
        # Set output path if not specified
        if output is None:
            output = Path(f"test_output_{voice_name}.wav")
        
        click.echo(f"🧪 Testing voice quality for '{voice_name}'")
        click.echo(f"📝 Test text: {text}")
        click.echo(f"📁 Output file: {output}")
        
        # Test the voice quality
        result = synthesizer.test_voice_quality(voice_name, text, output)
        
        # Display results
        click.echo(f"\n📊 Voice Quality Test Results:")
        click.echo(f"🎯 Overall Score: {result['quality_metrics']['overall_score']:.2f}/1.0")
        click.echo(f"🌿 Naturalness: {result['quality_metrics']['naturalness_score']:.2f}/1.0")
        click.echo(f"🎵 Pitch Quality: {result['quality_metrics']['pitch_score']:.2f}/1.0")
        click.echo(f"⏱️  Timing Quality: {result['quality_metrics']['timing_score']:.2f}/1.0")
        
        # Display feedback
        click.echo(f"\n💬 Quality Feedback:")
        for aspect, feedback in result['feedback'].items():
            click.echo(f"   {aspect.title()}: {feedback}")
        
        # Display recommendations
        if result['recommendations']:
            click.echo(f"\n💡 Improvement Recommendations:")
            for i, rec in enumerate(result['recommendations'], 1):
                click.echo(f"   {i}. {rec}")
        
        click.echo(f"\n✅ Voice quality test completed!")
        click.echo(f"🎵 Test audio saved to: {output}")
        click.echo(f"🔊 Listen to the audio to evaluate the improvements")
        
    except Exception as e:
        click.echo(f"❌ Voice quality test failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('text', type=str)
@click.option('--output', '-o', type=click.Path(path_type=Path), required=True,
               help='Output audio file path')
@click.option('--quality', '-q', type=click.Choice(['low', 'medium', 'high']), default='high',
               help='Voice quality level for maximum naturalness')
@click.option('--emotion', type=click.Choice(['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']),
               help='Emotional tone for natural inflection')
@click.option('--speed', type=click.FloatRange(0.5, 2.0), default=1.0,
               help='Speech rate (0.5 = slow, 2.0 = fast)')
@click.option('--pitch', type=click.IntRange(-12, 12), default=0,
               help='Pitch adjustment in semitones')
@click.option('--emphasis', type=click.Choice(['weak', 'normal', 'strong']), default='normal',
               help='Word emphasis level')
@click.pass_context
def enhanced_speak(ctx, text, output, quality, emotion, speed, pitch, emphasis):
    """Generate speech with enhanced naturalness and inflection."""
    config = ctx.obj['config']
    
    try:
        synthesizer = VoiceSynthesizer(config)
        
        # Set maximum quality for enhanced synthesis
        synthesizer.set_voice_quality(quality)
        
        # Enhanced parameters for maximum naturalness
        kwargs = {
            'quality': quality,
            'emotion': emotion,
            'speed': speed,
            'pitch': pitch,
            'emphasis': emphasis,
            'fix_speed': True
        }
        
        click.echo(f"🎤 Generating enhanced speech with maximum naturalness...")
        click.echo(f"🔧 Quality: {quality}")
        click.echo(f"😊 Emotion: {emotion}")
        click.echo(f"⚡ Speed: {speed}x")
        click.echo(f"🎵 Pitch: {pitch:+d} semitones")
        click.echo(f"💪 Emphasis: {emphasis}")
        
        output_path = synthesizer.text_to_speech(text, output, "default", "coqui_tts", **kwargs)
        
        click.echo(f"✅ Enhanced speech generated successfully!")
        click.echo(f"📁 Output file: {output_path}")
        click.echo(f"💡 This should sound much more natural than regular TTS!")
        
    except Exception as e:
        click.echo(f"❌ Enhanced speech synthesis failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def models(ctx):
    """List available high-quality TTS models for better inflection."""
    config = ctx.obj['config']
    
    try:
        synthesizer = VoiceSynthesizer(config)
        
        click.echo("🎯 High-Quality TTS Models (Better Inflection):")
        click.echo("=" * 50)
        
        models_list = synthesizer.get_available_models()
        if models_list:
            click.echo("📋 Available models (ranked by quality):")
            for i, model in enumerate(models_list, 1):
                # Highlight the best models
                if 'vits' in model.lower():
                    click.echo(f"  {i}. 🌟 {model} (Best inflection)")
                elif 'fast_pitch' in model.lower():
                    click.echo(f"  {i}. ⭐ {model} (Good prosody)")
                elif 'your_tts' in model.lower():
                    click.echo(f"  {i}. 🎭 {model} (Voice cloning)")
                else:
                    click.echo(f"  {i}. • {model}")
            
            click.echo("\n💡 Tip: Use VITS models for the most natural inflection!")
        else:
            click.echo("  No models available")
            
    except Exception as e:
        click.echo(f"❌ Failed to get models: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def trained_voices(ctx):
    """List available trained voice models."""
    config = ctx.obj['config']
    
    try:
        synthesizer = VoiceSynthesizer(config)
        
        click.echo("🎭 Trained Voice Models:")
        click.echo("=" * 50)
        
        voices_list = synthesizer.list_trained_voices()
        if voices_list:
            click.echo(f"📋 Found {len(voices_list)} trained voice models:")
            for i, voice in enumerate(voices_list, 1):
                name = voice.get('name', 'Unknown')
                engine = voice.get('engine', 'Unknown')
                quality = voice.get('quality', 'Unknown')
                files = voice.get('training_files', 'Unknown')
                
                click.echo(f"  {i}. 🎤 {name}")
                click.echo(f"     Engine: {engine}")
                click.echo(f"     Quality: {quality}")
                click.echo(f"     Training Files: {files}")
                click.echo()
            
            click.echo("💡 Use these voices with: speechscribe speak 'text' --voice <voice_name> --output file.wav")
        else:
            click.echo("  No trained voice models found")
            click.echo("  Train a voice with: speechscribe train-voice <folder> --output <voice_name>")
            
    except Exception as e:
        click.echo(f"❌ Failed to list trained voices: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('text', type=str)
@click.option('--output', '-o', type=click.Path(path_type=Path), required=True,
               help='Output audio file path')
@click.option('--voice', '-v', default='default', help='Voice name to use')
@click.option('--engine', '-e', default='coqui_tts',
               type=click.Choice(['coqui_tts', 'elevenlabs', 'azure_speech']),
               help='TTS engine to use')
@click.option('--emotion', type=click.Choice(['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised', 'Neutral']),
               default='Happy', help='Emotional tone for maximum naturalness')
@click.option('--emphasis', type=click.Choice(['weak', 'normal', 'moderate', 'strong']), default='strong',
               help='Emphasis level for natural expression')
@click.option('--speed', type=click.FloatRange(0.5, 2.0), default=1.0,
               help='Speech rate (0.5 = slow, 2.0 = fast)')
@click.option('--pitch', type=click.IntRange(-12, 12), default=0,
               help='Pitch adjustment in semitones')
@click.option('--quality', '-q', type=click.Choice(['low', 'medium', 'high']), default='high',
               help='Voice quality level for maximum naturalness')
@click.pass_context
def expressive_speak(ctx, text, output, voice, engine, emotion, emphasis, speed, pitch, quality):
    """Create truly expressive speech with maximum naturalness and emotional expression.
    
    This command uses advanced text processing and TTS parameters to create
    speech that sounds completely natural, emotionally expressive, and engaging.
    """
    try:
        # Initialize voice synthesizer
        synthesizer = VoiceSynthesizer(ctx.obj['config'])
        
        # Set voice quality
        synthesizer.set_voice_quality(quality)
        
        # Create expressive speech
        output_path = synthesizer.create_expressive_speech(
            text=text,
            output_path=output,
            voice_name=voice,
            engine=engine,
            emotion=emotion,
            emphasis=emphasis,
            speed=speed,
            pitch=pitch,
            quality=quality
        )
        
        click.echo(f"🎭 Expressive speech generated successfully!")
        click.echo(f"📁 Output: {output_path}")
        click.echo(f"🎨 Emotion: {emotion}")
        click.echo(f"💪 Emphasis: {emphasis}")
        click.echo(f"⚡ Speed: {speed}")
        click.echo(f"🎵 Pitch: {pitch}")
        click.echo(f"✨ Quality: {quality}")
        
    except Exception as e:
        click.echo(f"❌ Error creating expressive speech: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--recursive', '-r', is_flag=True, help='Process subdirectories recursively')
@click.option('--pattern', '-p', default='*.mp3,*.m4a,*.wav,*.flac,*.mov,*.mp4', 
               help='File pattern to match')
@click.option('--min-score', type=float, default=0.0, 
               help='Minimum quality score to display (0.0 to 1.0)')
def analyze_audio(input_path, recursive, pattern, min_score):
    """Analyze audio quality for voice training suitability."""
    try:
        # Initialize voice synthesizer
        config = Config()
        voice_synthesizer = VoiceSynthesizer(config)
        
        # Find audio files
        audio_files = find_audio_files(input_path, recursive, pattern)
        
        if not audio_files:
            click.echo("❌ No audio files found")
            return
        
        click.echo(f"🔍 Analyzing {len(audio_files)} audio files for voice training quality...")
        click.echo(f"📊 Quality threshold: {min_score:.2f}\n")
        
        # Analyze each file
        quality_results = []
        for audio_file in audio_files:
            quality = voice_synthesizer._analyze_audio_quality(audio_file)
            if quality['score'] >= min_score:
                quality_results.append((audio_file, quality))
        
        # Sort by quality score (highest first)
        quality_results.sort(key=lambda x: x[1]['score'], reverse=True)
        
        if not quality_results:
            click.echo(f"❌ No files meet the minimum quality score of {min_score:.2f}")
            return
        
        # Display results
        click.echo(f"✅ {len(quality_results)} files meet quality requirements:\n")
        
        for i, (audio_file, quality) in enumerate(quality_results, 1):
            score = quality['score']
            factors = quality.get('factors', [])
            
            # Color code the score
            if score >= 0.8:
                score_icon = "🟢"
                score_desc = "Excellent"
            elif score >= 0.6:
                score_icon = "🟡"
                score_desc = "Good"
            elif score >= 0.4:
                score_icon = "🟠"
                score_desc = "Fair"
            else:
                score_icon = "🔴"
                score_desc = "Poor"
            
            click.echo(f"{i:02d}. {score_icon} {audio_file.name}")
            click.echo(f"    📊 Score: {score:.2f}/1.0 ({score_desc})")
            click.echo(f"    🎵 Sample Rate: {quality.get('sample_rate', 'Unknown')} Hz")
            click.echo(f"    🔢 Bit Depth: {quality.get('bit_depth', 'Unknown')}-bit")
            click.echo(f"    ⏱️ Duration: {quality.get('duration', 0):.1f}s")
            click.echo(f"    🔊 Channels: {quality.get('channels', 'Unknown')}")
            click.echo(f"    🎛️ Codec: {quality.get('codec', 'Unknown')}")
            
            if factors:
                click.echo(f"    ✨ Factors: {', '.join(factors)}")
            
            click.echo()
        
        # Summary statistics
        scores = [q[1]['score'] for q in quality_results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score_actual = min(scores)
        
        click.echo(f"📈 Quality Summary:")
        click.echo(f"   • Average Score: {avg_score:.2f}/1.0")
        click.echo(f"   • Highest Score: {max_score:.2f}/1.0")
        click.echo(f"   • Lowest Score: {min_score_actual:.2f}/1.0")
        click.echo(f"   • Total Files: {len(quality_results)}")
        
        # Recommendations
        click.echo(f"\n💡 Recommendations:")
        if avg_score >= 0.8:
            click.echo(f"   🎉 Excellent audio quality! Use 'ultra' quality for maximum realism.")
        elif avg_score >= 0.6:
            click.echo(f"   👍 Good audio quality. Use 'high' quality for best results.")
        elif avg_score >= 0.4:
            click.echo(f"   ⚠️ Fair audio quality. Consider 'medium' quality or better source files.")
        else:
            click.echo(f"   ❌ Poor audio quality. Consider using higher quality source files.")
        
        click.echo(f"\n🚀 To train a voice model with these files:")
        click.echo(f"   speechscribe train-voice {input_path} --output voice_name --quality high")
        
    except Exception as e:
        click.echo(f"❌ Audio analysis failed: {e}")
        raise click.Abort()

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
