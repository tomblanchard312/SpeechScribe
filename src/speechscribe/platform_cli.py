"""
Platform CLI for SpeechScribe

New CLI that supports the modular platform architecture with profiles,
multiple engines, and various deployment modes.
"""

import logging
import sys
from pathlib import Path

import click

from .cli_utils import handle_cli_error
from .config import Config
from .control import (
    EngineRegistry,
    Environment,
    ProfileRegistry,
    RecommendationEngine,
)
from .orchestrator import SpeechScribeOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("speechscribe.log", encoding="utf-8"),
    ],
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
@click.version_option(version="2.0.0", prog_name="SpeechScribe Platform")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors")
@click.pass_context
def platform_cli(ctx, verbose, quiet):
    """SpeechScribe Platform - Modular Speech Intelligence

    Next-generation speech processing with profiles, multiple engines,
    and support for cloud, offline, and local deployments.
    """
    setup_logging(verbose, quiet)
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config()


@platform_cli.command()
@click.argument("profile", type=str)
@click.argument("source_type", type=click.Choice(["file", "teams", "zoom"]))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for file delivery",
)
@click.option("--webhook-url", type=str, help="Webhook URL for real-time delivery")
@click.option("--live-captions", type=str, help="URL for live caption streaming")
@click.option("--tts", is_flag=True, help="Enable text-to-speech audio generation")
@click.option(
    "--file-path",
    type=click.Path(exists=True, path_type=Path),
    help="Audio file path (for file source)",
)
@click.option("--meeting-url", type=str, help="Meeting URL (for teams/zoom sources)")
@click.pass_context
def process(
    ctx,
    profile,
    source_type,
    output_dir,
    webhook_url,
    live_captions,
    tts,
    file_path,
    meeting_url,
):
    """Process audio with the specified profile and source."""
    config = ctx.obj["config"]
    orchestrator = SpeechScribeOrchestrator(config.__dict__)

    # Validate profile
    profile_registry = ProfileRegistry()
    if profile not in profile_registry.list_profiles():
        click.echo(f"Unknown profile: {profile}")
        click.echo(f"Available profiles: {', '.join(profile_registry.list_profiles())}")
        return

    # Build source kwargs
    source_kwargs = {}
    if source_type == "file" and file_path:
        source_kwargs["file_paths"] = [file_path]
    elif source_type in ["teams", "zoom"] and meeting_url:
        source_kwargs["meeting_url" if source_type == "teams" else "meeting_id"] = (
            meeting_url
        )
        # TODO: Add credentials handling
        source_kwargs["credentials"] = {}  # Placeholder
    else:
        click.echo(f"Missing required parameters for {source_type} source")
        return

    # Build delivery config
    delivery_config = {}
    if output_dir:
        delivery_config["output_dir"] = str(output_dir)
    if webhook_url:
        delivery_config["webhook_url"] = webhook_url
    if live_captions:
        delivery_config["live_caption_url"] = live_captions
    if tts:
        delivery_config["tts_enabled"] = True
        if output_dir:
            delivery_config["tts_output_dir"] = str(output_dir)

    try:
        session_id = orchestrator.process_session(
            profile_name=profile,
            source_type=source_type,
            delivery_config=delivery_config,
            **source_kwargs,
        )
        click.echo(f"Processing completed. Session ID: {session_id}")
    except Exception as e:
        handle_cli_error(
            logger,
            "Processing failed. See speechscribe.log for details.",
            e,
        )


@platform_cli.command()
def list_profiles():
    """List available processing profiles."""
    profile_registry = ProfileRegistry()

    click.echo("Available Profiles:")
    click.echo("-" * 50)

    for profile_name in profile_registry.list_profiles():
        profile = profile_registry.get_profile(profile_name)
        click.echo(f"• {profile.name}")
        click.echo(f"  {profile.description}")
        click.echo(f"  Latency: {profile.latency_requirement.value}")
        click.echo(f"  Streaming: {profile.streaming_required}")
        click.echo(f"  Diarization: {profile.diarization_required}")
        click.echo(f"  Translation: {profile.translation_required}")
        if profile.translation_languages:
            click.echo(f"  Languages: {', '.join(profile.translation_languages)}")
        click.echo()


@platform_cli.command()
@click.argument("profile", type=str)
def recommend(profile):
    """Show recommendations for a profile."""
    try:
        recommender = RecommendationEngine()
        rec_config = recommender.recommend_configuration(profile)

        click.echo(f"Profile: {profile}")
        click.echo(f"Recommended Engine: {rec_config['engine']}")
        click.echo(f"Environment: {rec_config['environment'].value}")
        click.echo(
            f"Capabilities: {rec_config['capabilities'].streaming_support and 'Streaming' or 'Batch'}"
        )

    except ValueError as e:
        handle_cli_error(
            logger,
            "Recommendation failed. See speechscribe.log for details.",
            e,
        )


@platform_cli.command()
def list_engines():
    """List available speech processing engines."""
    engine_registry = EngineRegistry()

    click.echo("Available Engines:")
    click.echo("-" * 50)

    for env in [Environment.AZURE, Environment.OFFLINE, Environment.LOCAL]:
        click.echo(f"\n{env.value.upper()} Environment:")
        engines = engine_registry.get_available_engines(env)
        if engines:
            for engine in engines:
                caps = engine_registry.engines[engine]
                features = []
                if caps.streaming_support:
                    features.append("Streaming")
                if caps.diarization_support:
                    features.append("Diarization")
                if caps.translation_support:
                    features.append("Translation")
                if caps.tts_support:
                    features.append("TTS")

                click.echo(f"  • {engine}: {', '.join(features)}")
        else:
            click.echo("  No engines available")


# Legacy CLI commands for backward compatibility
@platform_cli.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    "-m",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v3"]),
    help="Whisper model size (legacy)",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["cpu", "cuda"]),
    help="Inference device (legacy)",
)
@click.option(
    "--output-dir", "-o", type=click.Path(path_type=Path), help="Output directory"
)
@click.pass_context
def transcribe_legacy(ctx, audio_file, model, device, output_dir):
    """Legacy transcription command (use 'process' for new features)."""
    from .core import TranscriptionManager

    config = ctx.obj["config"]

    # Override config with CLI options
    if model:
        config.set("model", model)
    if device:
        config.set("device", device)

    try:
        manager = TranscriptionManager(config)
        output_files = manager.process_single_file(audio_file, output_dir)
        click.echo(f"Generated {len(output_files)} output files")
    except Exception as e:
        handle_cli_error(
            logger,
            "Transcription failed. See speechscribe.log for details.",
            e,
        )


if __name__ == "__main__":
    platform_cli()
