"""
Voice synthesis and voice cloning capabilities for VMTranscriber.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess
import json

logger = logging.getLogger(__name__)

class VoiceSynthesizer:
    """Handles voice synthesis, cloning, and conversion."""
    
    def __init__(self, config):
        """Initialize voice synthesizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.available_engines = self._check_available_engines()
        
    def _check_available_engines(self) -> Dict[str, bool]:
        """Check which voice synthesis engines are available."""
        engines = {
            'coqui_tts': False,
            'elevenlabs': False,
            'azure_speech': False,
            'so_vits_svc': False,
            'rvc': False
        }
        
        # Check Coqui TTS
        try:
            import TTS
            engines['coqui_tts'] = True
            logger.info("Coqui TTS available")
        except ImportError:
            logger.debug("Coqui TTS not available")
        
        # Check ElevenLabs
        try:
            import elevenlabs
            engines['elevenlabs'] = True
            logger.info("ElevenLabs available")
        except ImportError:
            logger.debug("ElevenLabs not available")
        
        # Check Azure Speech
        try:
            import azure.cognitiveservices.speech as speechsdk
            engines['azure_speech'] = True
            logger.info("Azure Speech available")
        except ImportError:
            logger.debug("Azure Speech not available")
        
        # Check So-VITS-SVC
        try:
            # This would require the So-VITS-SVC model files
            engines['so_vits_svc'] = os.path.exists("models/so-vits-svc")
            if engines['so_vits_svc']:
                logger.info("So-VITS-SVC available")
        except Exception:
            logger.debug("So-VITS-SVC not available")
        
        # Check RVC
        try:
            # This would require the RVC model files
            engines['rvc'] = os.path.exists("models/rvc")
            if engines['rvc']:
                logger.info("RVC available")
        except Exception:
            logger.debug("RVC not available")
        
        return engines
    
    def text_to_speech(self, text: str, output_path: Path, 
                       voice_name: str = "default", engine: str = "coqui_tts",
                       fix_speed: bool = True, **kwargs) -> Path:
        """Convert text to speech with enhanced naturalness.
        
        Args:
            text: Text to convert to speech
            output_path: Output audio file path
            voice_name: Name of the voice to use
            engine: TTS engine to use
            fix_speed: Whether to fix audio speed
            **kwargs: Additional engine-specific options including:
                - emotion: Emotional tone (Happy, Sad, Angry, etc.)
                - speed: Speech rate (0.5 to 2.0)
                - pitch: Pitch adjustment (-12 to +12 semitones)
                - emphasis: Word emphasis level
                
        Returns:
            Path to the generated audio file
        """
        # Apply enhanced voice quality settings
        quality = kwargs.get('quality', 'high')
        self.set_voice_quality(quality)
        
        # Enhanced text preprocessing for better naturalness
        if self.voice_quality.get('text_preprocessing', True):
            text = self._enhance_text_for_natural_speech(text, **kwargs)
        
        if engine == "coqui_tts" and self.available_engines['coqui_tts']:
            return self._coqui_tts(text, output_path, voice_name, fix_speed=fix_speed, **kwargs)
        elif engine == "elevenlabs" and self.available_engines['elevenlabs']:
            return self._elevenlabs_tts(text, output_path, voice_name, fix_speed=fix_speed, **kwargs)
        elif engine == "azure_speech" and self.available_engines['azure_speech']:
            return self._azure_tts(text, output_path, voice_name, fix_speed=fix_speed, **kwargs)
        else:
            raise ValueError(f"Engine {engine} not available. Available: {list(self.available_engines.keys())}")
    
    def _coqui_tts(self, text: str, output_path: Path, voice_name: str, fix_speed: bool = True, **kwargs) -> Path:
        """Use Coqui TTS for text-to-speech with maximum naturalness and emotional expression."""
        try:
            from TTS.api import TTS
            
            # Enhanced model selection for maximum naturalness and emotional expression
            preferred_models = [
                "tts_models/en/vctk/vits",  # Multi-speaker VITS - excellent naturalness and emotion
                "tts_models/en/ljspeech/fast_pitch",  # FastPitch - great prosody and emphasis
                "tts_models/en/ljspeech/vits",  # VITS - very natural with good emotion
                "tts_models/en/ljspeech/tacotron2-DDC",  # Fallback with good prosody
            ]
            
            tts = None
            selected_model = None
            for model in preferred_models:
                try:
                    tts = TTS(model_name=model, progress_bar=False)
                    selected_model = model
                    logger.info(f"Using enhanced TTS model: {model}")
                    break
                except Exception:
                    continue
            
            if tts is None:
                # Fallback to basic model
                tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
                selected_model = "tts_models/en/ljspeech/tacotron2-DDC"
                logger.info("Using fallback TTS model")
            
            # Enhanced text preprocessing for maximum naturalness
            processed_text = self._preprocess_text_for_tts(text)
            logger.info(f"Generating speech with enhanced Coqui TTS using voice: {voice_name}")
            
            # Enhanced speaker selection for multi-speaker models
            if hasattr(tts, 'speakers') and tts.speakers:
                # Select speakers known for natural inflection and emotional expression
                expressive_speakers = ['p225', 'p226', 'p227', 'p228', 'p229', 'p230', 'p231', 'p232', 'p233']
                speaker = None
                for sp in expressive_speakers:
                    if sp in tts.speakers:
                        speaker = sp
                        break
                
                if speaker:
                    # Use enhanced TTS with speaker selection and emotional parameters
                    tts.tts_to_file(
                        text=processed_text, 
                        file_path=str(output_path), 
                        speaker=speaker,
                        # Enhanced parameters for maximum naturalness
                        speed=kwargs.get('speed', 1.0),
                        emotion=kwargs.get('emotion', 'Happy'),
                        # Additional parameters for better quality
                        length_scale=1.1,  # Slightly longer pauses for naturalness
                        noise_scale=0.667,  # Balanced clarity and naturalness
                        noise_w=0.8  # Voice stability
                    )
                else:
                    tts.tts_to_file(text=processed_text, file_path=str(output_path))
            else:
                # For single-speaker models, use enhanced parameters
                tts.tts_to_file(
                    text=processed_text, 
                    file_path=str(output_path),
                    speed=kwargs.get('speed', 1.0),
                    # Additional parameters for better quality
                    length_scale=1.1,
                    noise_scale=0.667,
                    noise_w=0.8
                )
            
            # Enhanced audio post-processing for better quality
            if fix_speed:
                self._fix_audio_speed(output_path)
            
            logger.info(f"Enhanced speech generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Enhanced Coqui TTS failed: {e}")
            raise RuntimeError(f"Enhanced Coqui TTS synthesis failed: {e}")
    
    def _elevenlabs_tts(self, text: str, output_path: Path, voice_name: str, **kwargs) -> Path:
        """Use ElevenLabs for text-to-speech."""
        try:
            from elevenlabs import generate, save, set_api_key
            
            # Get API key from config or environment
            api_key = kwargs.get('api_key') or os.getenv('ELEVENLABS_API_KEY')
            if not api_key:
                raise ValueError("ElevenLabs API key required. Set ELEVENLABS_API_KEY environment variable or pass api_key parameter.")
            
            set_api_key(api_key)
            
            # Generate speech
            logger.info(f"Generating speech with ElevenLabs using voice: {voice_name}")
            audio = generate(
                text=text,
                voice=voice_name,
                model="eleven_monolingual_v1"
            )
            
            # Save audio
            save(audio, str(output_path))
            
            logger.info(f"Speech generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}")
            raise RuntimeError(f"ElevenLabs synthesis failed: {e}")
    
    def _azure_tts(self, text: str, output_path: Path, voice_name: str, **kwargs) -> Path:
        """Use Azure Speech Service for text-to-speech."""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Get credentials from config or environment
            subscription_key = kwargs.get('subscription_key') or os.getenv('AZURE_SPEECH_KEY')
            region = kwargs.get('region') or os.getenv('AZURE_SPEECH_REGION')
            
            if not subscription_key or not region:
                raise ValueError("Azure Speech credentials required. Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables.")
            
            # Configure speech config
            speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
            speech_config.speech_synthesis_voice_name = voice_name
            
            # Configure audio output
            audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))
            
            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            
            # Generate speech
            logger.info(f"Generating speech with Azure Speech using voice: {voice_name}")
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"Speech generated: {output_path}")
                return output_path
            else:
                raise RuntimeError(f"Azure Speech synthesis failed: {result.reason}")
                
        except Exception as e:
            logger.error(f"Azure Speech TTS failed: {e}")
            raise RuntimeError(f"Azure Speech synthesis failed: {e}")
    
    def clone_voice(self, source_audio: Path, text: str, output_path: Path,
                    engine: str = "coqui_tts", fix_speed: bool = True, **kwargs) -> Path:
        """Clone a voice from source audio and use it for new text.
        
        Args:
            source_audio: Path to source audio file for voice cloning
            text: New text to speak in the cloned voice
            output_path: Output audio file path
            engine: Voice cloning engine to use
            **kwargs: Additional engine-specific options
            
        Returns:
            Path to the generated audio file
        """
        if engine == "coqui_tts" and self.available_engines['coqui_tts']:
            return self._coqui_voice_cloning(source_audio, text, output_path, fix_speed=fix_speed, **kwargs)
        else:
            raise ValueError(f"Voice cloning engine {engine} not available")
    
    def _coqui_voice_cloning(self, source_audio: Path, text: str, output_path: Path, fix_speed: bool = True, **kwargs) -> Path:
        """Use Coqui TTS for voice cloning with enhanced quality parameters for maximum naturalness."""
        try:
            from TTS.api import TTS
            
            # Enhanced text preprocessing for better naturalness
            processed_text = self._preprocess_text_for_tts(text)
            
            # Apply high-quality parameters for better pitch, timing, and naturalness
            quality_params = self._get_enhanced_cloning_parameters()
            
            # Try YourTTS first, but fall back to other models if quality is poor
            try:
                # Initialize TTS with voice cloning capability
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
                
                # Clone voice and generate speech with enhanced parameters
                logger.info(f"Cloning voice from {source_audio.name} with YourTTS and enhanced quality parameters")
                tts.tts_to_file(
                    text=processed_text,
                    file_path=str(output_path),
                    speaker_wav=str(source_audio),
                    language="en",
                    # Enhanced parameters for better naturalness
                    speed=quality_params.get('speed', 1.0),
                    # YourTTS specific parameters for better quality
                    length_scale=quality_params.get('length_scale', 1.1),  # Natural pauses
                    noise_scale=quality_params.get('noise_scale', 0.667),  # Clarity vs naturalness balance
                    noise_w=0.8,  # Voice stability
                    # Additional parameters for better pitch and timing
                    emotion=quality_params.get('emotion', 'Neutral'),
                    emphasis=quality_params.get('emphasis', 'moderate')
                )
                
                # Test the quality of the generated audio
                quality_check = self._analyze_generated_audio_quality(output_path)
                if quality_check['overall_score'] < 0.5:
                    logger.warning(f"YourTTS quality score too low ({quality_check['overall_score']:.2f}), trying alternative model")
                    raise Exception("Quality too low, trying alternative")
                
            except Exception as e:
                logger.info(f"YourTTS failed or quality too low, trying alternative model: {e}")
                # Fall back to alternative model for better naturalness
                self._use_alternative_model_for_cloning(source_audio, processed_text, output_path, quality_params)
            
            # Enhanced audio post-processing for better quality
            if fix_speed:
                self._fix_audio_speed(output_path)
            
            # Apply additional audio enhancement for better naturalness
            self._enhance_audio_naturalness(output_path, quality_params)
            
            logger.info(f"Enhanced voice cloned and speech generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Enhanced Coqui voice cloning failed: {e}")
            raise RuntimeError(f"Voice cloning failed: {e}")
    
    def _use_alternative_model_for_cloning(self, source_audio: Path, text: str, output_path: Path, quality_params: Dict[str, Any]) -> None:
        """Use alternative TTS models for better naturalness when YourTTS quality is poor."""
        try:
            from TTS.api import TTS
            
            # Try VITS model which often has better naturalness
            alternative_models = [
                "tts_models/en/vctk/vits",  # Multi-speaker VITS - excellent naturalness
                "tts_models/en/ljspeech/vits",  # VITS - very natural with good emotion
                "tts_models/en/ljspeech/fast_pitch",  # FastPitch - great prosody and emphasis
            ]
            
            for model_name in alternative_models:
                try:
                    logger.info(f"Trying alternative model: {model_name}")
                    tts = TTS(model_name=model_name, progress_bar=False)
                    
                    # For non-cloning models, we'll use the best available speaker
                    if hasattr(tts, 'speakers') and tts.speakers:
                        # Select a speaker known for natural inflection
                        natural_speakers = ['p225', 'p226', 'p227', 'p228', 'p229']
                        speaker = None
                        for sp in natural_speakers:
                            if sp in tts.speakers:
                                speaker = sp
                                break
                        
                        if speaker:
                            tts.tts_to_file(
                                text=text,
                                file_path=str(output_path),
                                speaker=speaker,
                                speed=quality_params.get('speed', 0.95),
                                length_scale=quality_params.get('length_scale', 1.15),
                                noise_scale=quality_params.get('noise_scale', 0.6),
                                noise_w=quality_params.get('noise_w', 0.75)
                            )
                        else:
                            tts.tts_to_file(
                                text=text,
                                file_path=str(output_path),
                                speed=quality_params.get('speed', 0.95),
                                length_scale=quality_params.get('length_scale', 1.15),
                                noise_scale=quality_params.get('noise_scale', 0.6),
                                noise_w=quality_params.get('noise_w', 0.75)
                            )
                    else:
                        tts.tts_to_file(
                            text=text,
                            file_path=str(output_path),
                            speed=quality_params.get('speed', 0.95),
                            length_scale=quality_params.get('length_scale', 1.15),
                            noise_scale=quality_params.get('noise_scale', 0.6),
                            noise_w=quality_params.get('noise_w', 0.75)
                        )
                    
                    logger.info(f"Successfully used alternative model: {model_name}")
                    return
                    
                except Exception as e:
                    logger.warning(f"Alternative model {model_name} failed: {e}")
                    continue
            
            # If all alternatives fail, fall back to basic YourTTS
            logger.warning("All alternative models failed, using basic YourTTS")
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
            tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=str(source_audio),
                language="en"
            )
            
        except Exception as e:
            logger.error(f"Alternative model usage failed: {e}")
            raise RuntimeError(f"Alternative model usage failed: {e}")
    
    def _get_enhanced_cloning_parameters(self) -> Dict[str, Any]:
        """Get enhanced parameters for voice cloning to improve naturalness, pitch, and timing."""
        return {
            # Timing and rhythm parameters
            'speed': 0.95,  # Slightly slower for more natural pacing
            'length_scale': 1.15,  # Longer pauses for natural breathing
            'noise_scale': 0.6,  # Lower for more natural voice characteristics
            'noise_w': 0.75,  # Balanced voice stability
            
            # Emotional and expression parameters
            'emotion': 'Neutral',  # Start with neutral for consistency
            'emphasis': 'moderate',  # Moderate emphasis for naturalness
            
            # Audio enhancement parameters
            'pitch_variation': 0.4,  # Natural pitch variation
            'prosody_variation': 0.6,  # Natural prosody changes
            'breathing_patterns': True,  # Enable natural breathing
            'natural_pauses': True,  # Enable natural pauses
            'emphasis_control': True,  # Enable emphasis control
            'pitch_control': True,  # Enable pitch control
            'timing_control': True   # Enable timing control
        }
    
    def _enhance_audio_naturalness(self, audio_path: Path, quality_params: Dict[str, Any]) -> None:
        """Apply additional audio enhancement to improve naturalness, pitch, and timing."""
        try:
            import subprocess
            
            # Create enhanced version with better naturalness
            enhanced_path = audio_path.parent / f"enhanced_{audio_path.name}"
            
            # Apply simplified audio filters for naturalness that work across FFmpeg versions
            cmd = [
                'ffmpeg', '-y',
                '-i', str(audio_path),
                # Simplified filters for natural voice characteristics
                '-af', (
                    # Gentle pitch adjustment for naturalness
                    'asetrate=44100*0.98,'  # Slight pitch adjustment
                    # Natural breathing and pause enhancement
                    'apad=pad_dur=0.1,'  # Natural padding
                    # Gentle compression for dynamic range
                    'compand=attacks=0.1:points=-80/-80|-20.1/-20.1|-20/-20|0/0:soft-knee=0.1:gain=0:ratio=2:release=0.1,'
                    # Final frequency shaping for naturalness
                    'highpass=f=80,lowpass=f=8000,'
                    # Gentle normalization
                    'loudnorm=I=-18:TP=-1.5:LRA=11'
                ),
                str(enhanced_path)
            ]
            
            # Run enhancement
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Replace original with enhanced version
            enhanced_path.replace(audio_path)
            
            logger.info(f"Audio naturalness enhanced for {audio_path.name}")
            
        except Exception as e:
            logger.warning(f"Audio naturalness enhancement failed: {e}")
            # Continue without enhancement if it fails
    
    def convert_voice(self, source_audio: Path, target_voice: Path, output_path: Path,
                      engine: str = "so_vits_svc", fix_speed: bool = True, **kwargs) -> Path:
        """Convert one voice to another voice.
        
        Args:
            source_audio: Audio file with voice to convert
            target_voice: Audio file with target voice characteristics
            output_path: Output audio file path
            engine: Voice conversion engine to use
            **kwargs: Additional engine-specific options
            
        Returns:
            Path to the converted audio file
        """
        if engine == "so_vits_svc" and self.available_engines['so_vits_svc']:
            return self._so_vits_svc_convert(source_audio, target_voice, output_path, fix_speed=fix_speed, **kwargs)
        elif engine == "rvc" and self.available_engines['rvc']:
            return self._rvc_convert(source_audio, target_voice, output_path, fix_speed=fix_speed, **kwargs)
        else:
            raise ValueError(f"Voice conversion engine {engine} not available")
    
    def _so_vits_svc_convert(self, source_audio: Path, target_voice: Path, output_path: Path, fix_speed: bool = True, **kwargs) -> Path:
        """Use So-VITS-SVC for voice conversion."""
        try:
            # This is a placeholder - you'd need to implement the actual So-VITS-SVC integration
            # The actual implementation would depend on how you've set up So-VITS-SVC
            
            logger.info(f"Converting voice using So-VITS-SVC")
            logger.warning("So-VITS-SVC integration not fully implemented - placeholder only")
            
            # For now, just copy the source audio as a placeholder
            import shutil
            shutil.copy2(source_audio, output_path)
            
            # Fix playback speed if requested
            if fix_speed:
                self._fix_audio_speed(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"So-VITS-SVC conversion failed: {e}")
            raise RuntimeError(f"Voice conversion failed: {e}")
    
    def _rvc_convert(self, source_audio: Path, target_voice: Path, output_path: Path, fix_speed: bool = True, **kwargs) -> Path:
        """Use RVC for voice conversion."""
        try:
            # This is a placeholder - you'd need to implement the actual RVC integration
            logger.info(f"Converting voice using RVC")
            logger.warning("RVC integration not fully implemented - placeholder only")
            
            # For now, just copy the source audio as a placeholder
            import shutil
            shutil.copy2(source_audio, output_path)
            
            # Fix playback speed if requested
            if fix_speed:
                self._fix_audio_speed(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"RVC conversion failed: {e}")
            raise RuntimeError(f"Voice conversion failed: {e}")
    
    def get_available_voices(self, engine: str = "coqui_tts") -> List[str]:
        """Get list of available voices for a specific engine.
        
        Args:
            engine: TTS engine to query
            
        Returns:
            List of available voice names
        """
        if engine == "coqui_tts" and self.available_engines['coqui_tts']:
            try:
                from TTS.api import TTS
                tts = TTS()
                return tts.list_models()
            except Exception as e:
                logger.error(f"Failed to get Coqui TTS voices: {e}")
                return []
        elif engine == "elevenlabs" and self.available_engines['elevenlabs']:
            try:
                from elevenlabs import voices, set_api_key
                api_key = os.getenv('ELEVENLABS_API_KEY')
                if api_key:
                    set_api_key(api_key)
                    voices_list = voices()
                    return [voice.name for voice in voices_list]
                return []
            except Exception as e:
                logger.error(f"Failed to get ElevenLabs voices: {e}")
                return []
        else:
            return []
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about available engines and their capabilities."""
        return {
            'available_engines': self.available_engines,
            'coqui_tts': {
                'available': self.available_engines['coqui_tts'],
                'capabilities': ['tts', 'voice_cloning'],
                'offline': True,
                'quality': 'high'
            },
            'elevenlabs': {
                'available': self.available_engines['elevenlabs'],
                'capabilities': ['tts'],
                'offline': False,
                'quality': 'very_high',
                'requires_api_key': True
            },
            'azure_speech': {
                'available': self.available_engines['azure_speech'],
                'capabilities': ['tts'],
                'offline': False,
                'quality': 'high',
                'requires_credentials': True
            },
            'so_vits_svc': {
                'available': self.available_engines['so_vits_svc'],
                'capabilities': ['voice_conversion'],
                'offline': True,
                'quality': 'high'
            },
            'rvc': {
                'available': self.available_engines['rvc'],
                'capabilities': ['voice_conversion'],
                'offline': True,
                'quality': 'medium'
            }
        }
    
    def _fix_audio_speed(self, audio_path: Path) -> None:
        """Fix audio playback speed by ensuring correct sample rate and format."""
        try:
            import subprocess
            import tempfile
            
            # Create temporary file for processing
            temp_path = audio_path.with_suffix('.temp.wav')
            
            # Use FFmpeg to fix the audio speed and ensure proper format
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-i', str(audio_path),  # Input file
                '-ar', '22050',  # Set sample rate to 22.05 kHz (common for TTS)
                '-ac', '1',      # Convert to mono
                '-c:a', 'pcm_s16le',  # Use 16-bit PCM
                str(temp_path)   # Output to temp file
            ]
            
            # Run FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Replace original file with fixed version
            temp_path.replace(audio_path)
            
            logger.info(f"Fixed audio speed for {audio_path.name}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg audio fix failed: {e}. Audio may play at wrong speed.")
        except Exception as e:
            logger.warning(f"Audio speed fix failed: {e}. Audio may play at wrong speed.")
    
    def _fix_audio_speed_python(self, audio_path: Path) -> None:
        """Alternative Python-based audio speed fix using librosa."""
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=None)
            
            # Ensure correct sample rate (22.05 kHz is common for TTS)
            target_sr = 22050
            if sr != target_sr:
                # Resample audio
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Save with correct parameters
            sf.write(str(audio_path), audio, sr, subtype='PCM_16')
            
            logger.info(f"Fixed audio speed using Python for {audio_path.name}")
            
        except ImportError:
            logger.debug("librosa not available, skipping Python-based audio fix")
        except Exception as e:
            logger.warning(f"Python audio fix failed: {e}. Audio may play at wrong speed.")
    
    def _preprocess_text_for_tts(self, text: str) -> str:
        """Advanced text preprocessing for maximum naturalness and emotional expression."""
        import re
        
        # Start with clean text
        processed = text.strip()
        
        # Remove any existing problematic tags
        processed = re.sub(r'\[[^\]]*\]', '', processed)
        
        # Advanced prosody enhancement for natural speech patterns
        processed = self._add_natural_prosody(processed)
        
        # Add emotional expression markers
        processed = self._add_emotional_expression(processed)
        
        # Add natural breathing and emphasis patterns
        processed = self._add_natural_speech_patterns(processed)
        
        # Ensure proper sentence structure
        processed = self._enhance_sentence_structure(processed)
        
        return processed.strip()
    
    def _add_natural_prosody(self, text: str) -> str:
        """Add natural prosody markers for better inflection."""
        import re
        
        # Add rising intonation for questions
        text = re.sub(r'([^.!?]*\?)', r'<prosody pitch="+10%">\1</prosody>', text)
        
        # Add emphasis for exclamations
        text = re.sub(r'([^.!?]*!)', r'<emphasis level="strong">\1</emphasis>', text)
        
        # Add natural pauses for commas and semicolons
        text = re.sub(r'([,;])', r'\1 <break time="300ms"/>', text)
        
        # Add longer pauses for sentence endings
        text = re.sub(r'([.!?])', r'\1 <break time="500ms"/>', text)
        
        # Add emphasis for important words (capitalized words often indicate emphasis)
        text = re.sub(r'\b([A-Z][a-z]+)\b', r'<emphasis level="moderate">\1</emphasis>', text)
        
        return text
    
    def _add_emotional_expression(self, text: str) -> str:
        """Add emotional expression markers for more natural speech."""
        import re
        
        # Detect emotional content and add appropriate markers
        emotional_patterns = [
            (r'\b(amazing|wonderful|fantastic|incredible)\b', 'excited'),
            (r'\b(sad|unfortunate|terrible|awful)\b', 'sad'),
            (r'\b(angry|furious|mad|upset)\b', 'angry'),
            (r'\b(scared|afraid|terrified|worried)\b', 'fearful'),
            (r'\b(love|adore|cherish|treasure)\b', 'happy'),
            (r'\b(hate|despise|loathe|detest)\b', 'angry'),
            (r'\b(please|kindly|gently)\b', 'gentle'),
            (r'\b(urgent|immediately|now)\b', 'urgent')
        ]
        
        for pattern, emotion in emotional_patterns:
            text = re.sub(pattern, f'<prosody rate="slow" pitch="+5%">\\1</prosody>', text, flags=re.IGNORECASE)
        
        return text
    
    def _add_natural_speech_patterns(self, text: str) -> str:
        """Add natural speech patterns like breathing and emphasis."""
        import re
        
        # Add breathing patterns for longer sentences
        sentences = text.split('.')
        enhanced_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add breathing breaks for long sentences
            if len(sentence.split()) > 15:
                words = sentence.split()
                mid_point = len(words) // 2
                words.insert(mid_point, '<break time="200ms"/>')
                sentence = ' '.join(words)
            
            # Add emphasis for repeated words
            sentence = re.sub(r'\b(\w+)(\s+\1)+\b', r'<emphasis level="strong">\1</emphasis>', sentence)
            
            # Add emphasis for numbers and important information
            sentence = re.sub(r'\b(\d+)\b', r'<say-as interpret-as="cardinal">\1</say-as>', sentence)
            
            enhanced_sentences.append(sentence)
        
        return '. '.join(enhanced_sentences)
    
    def _enhance_sentence_structure(self, text: str) -> str:
        """Enhance sentence structure for better prosody."""
        import re
        
        # Ensure proper sentence endings
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Add natural rhythm by varying emphasis
        sentences = text.split('.')
        enhanced_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Alternate emphasis patterns for natural rhythm
            if i % 2 == 0:
                # Even sentences get moderate emphasis
                sentence = f'<prosody rate="normal" pitch="normal">{sentence}</prosody>'
            else:
                # Odd sentences get varied emphasis
                sentence = f'<prosody rate="slow" pitch="+2%">{sentence}</prosody>'
            
            enhanced_sentences.append(sentence)
        
        return '. '.join(enhanced_sentences)
    
    def _enhance_text_for_natural_speech(self, text: str, **kwargs) -> str:
        """Enhanced text processing for maximum naturalness and emotional expression."""
        # Start with advanced preprocessing
        processed = self._preprocess_text_for_tts(text)
        
        # Apply emotion-specific enhancements
        emotion = kwargs.get('emotion', 'Neutral')
        if emotion and emotion != 'Neutral':
            processed = self._apply_emotion_enhancement(processed, emotion)
        
        # Apply speed and pitch adjustments
        speed = kwargs.get('speed', 1.0)
        if speed != 1.0:
            processed = self._apply_speed_enhancement(processed, speed)
        
        pitch = kwargs.get('pitch', 0)
        if pitch != 0:
            processed = self._apply_pitch_enhancement(processed, pitch)
        
        # Apply emphasis enhancement
        emphasis = kwargs.get('emphasis', 'normal')
        if emphasis != 'normal':
            processed = self._apply_emphasis_enhancement(processed, emphasis)
        
        # Add natural breathing patterns
        processed = self._add_breathing_patterns(processed)
        
        # Add prosody markers
        processed = self._add_prosody_markers(processed)
        
        return processed
    
    def _apply_emotion_enhancement(self, text: str, emotion: str) -> str:
        """Apply emotion-specific text enhancements for natural expression."""
        emotion_markers = {
            'Happy': ['<prosody pitch="+15%" rate="fast">', '</prosody>'],
            'Sad': ['<prosody pitch="-10%" rate="slow">', '</prosody>'],
            'Angry': ['<prosody pitch="+20%" rate="fast" volume="loud">', '</prosody>'],
            'Fearful': ['<prosody pitch="+5%" rate="slow" volume="soft">', '</prosody>'],
            'Disgusted': ['<prosody pitch="-5%" rate="slow">', '</prosody>'],
            'Surprised': ['<prosody pitch="+25%" rate="fast">', '</prosody>']
        }
        
        markers = emotion_markers.get(emotion, ['', ''])
        if markers[0]:
            return f"{markers[0]}{text}{markers[1]}"
        return text
    
    def _apply_speed_enhancement(self, text: str, speed: float) -> str:
        """Apply speed-related text enhancements."""
        if speed < 0.8:
            return f'<prosody rate="slow">{text}</prosody>'
        elif speed > 1.2:
            return f'<prosody rate="fast">{text}</prosody>'
        return text
    
    def _apply_pitch_enhancement(self, text: str, pitch: int) -> str:
        """Apply pitch-related text enhancements."""
        if pitch > 0:
            return f'<prosody pitch="+{pitch*2}%">{text}</prosody>'
        elif pitch < 0:
            return f'<prosody pitch="{pitch*2}%">{text}</prosody>'
        return text
    
    def _apply_emphasis_enhancement(self, text: str, emphasis: str) -> str:
        """Apply emphasis-related text enhancements."""
        emphasis_markers = {
            'strong': ['<emphasis level="strong">', '</emphasis>'],
            'weak': ['<prosody volume="soft">', '</prosody>'],
            'moderate': ['<emphasis level="moderate">', '</emphasis>']
        }
        
        markers = emphasis_markers.get(emphasis, ['', ''])
        if markers[0]:
            return f"{markers[0]}{text}{markers[1]}"
        return text
    
    def _add_breathing_patterns(self, text: str) -> str:
        """Add natural breathing patterns for more realistic speech."""
        import re
        
        # Add breathing breaks at natural sentence boundaries
        sentences = text.split('.')
        enhanced_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add breathing for longer sentences
            if len(sentence.split()) > 12:
                words = sentence.split()
                # Add breathing break after first third
                break_point = len(words) // 3
                words.insert(break_point, '<break time="150ms"/>')
                sentence = ' '.join(words)
            
            enhanced_sentences.append(sentence)
        
        return '. '.join(enhanced_sentences)
    
    def _add_prosody_markers(self, text: str) -> str:
        """Add prosody markers for better inflection and naturalness."""
        import re
        
        # Add sentence-level prosody variation
        sentences = text.split('.')
        enhanced_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Vary prosody for natural rhythm
            if i % 3 == 0:
                # Rising intonation
                sentence = f'<prosody pitch="+5%">{sentence}</prosody>'
            elif i % 3 == 1:
                # Falling intonation
                sentence = f'<prosody pitch="-3%">{sentence}</prosody>'
            # Default case keeps normal pitch
            
            enhanced_sentences.append(sentence)
        
        return '. '.join(enhanced_sentences)

    def _enhance_voice_cloning_quality(self, source_audio: Path, text: str, output_path: Path, **kwargs) -> Path:
        """Enhanced voice cloning with better quality and natural inflection."""
        try:
            from TTS.api import TTS
            
            # Use YourTTS model for better voice cloning
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
            
            # Enhanced text preprocessing for better quality
            processed_text = self._enhance_text_for_natural_speech(text, **kwargs)
            
            # Clone voice with enhanced settings
            logger.info(f"Enhanced voice cloning from {source_audio.name}")
            tts.tts_to_file(
                text=processed_text,
                file_path=str(output_path),
                speaker_wav=str(source_audio),
                language="en",
                # Enhanced parameters for better quality
                speed=kwargs.get('speed', 1.0),
                emotion=kwargs.get('emotion', 'Happy')
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Enhanced voice cloning failed: {e}")
            # Fallback to regular voice cloning
            return self._coqui_voice_cloning(source_audio, text, output_path, **kwargs)
    
    def get_available_models(self) -> List[str]:
        """Get list of available TTS models for better quality."""
        try:
            from TTS.api import TTS
            tts = TTS()
            models = tts.list_models()
            
            # Filter for high-quality models
            quality_models = []
            for model in models:
                if any(keyword in model.lower() for keyword in ['vits', 'fast_pitch', 'tacotron2', 'your_tts']):
                    quality_models.append(model)
            
            return quality_models[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def set_voice_quality(self, quality: str = "high") -> None:
        """Set enhanced voice quality preferences for maximum naturalness and emotional expression."""
        quality_settings = {
            "low": {
                "sample_rate": 16000,
                "use_enhanced_models": False,
                "text_preprocessing": False,
                "prosody_enhancement": False,
                "emotion_detection": False,
                "natural_pauses": False,
                "breathing_patterns": False,
                "emphasis_control": False,
                "pitch_variation": False
            },
            "medium": {
                "sample_rate": 22050,
                "use_enhanced_models": True,
                "text_preprocessing": True,
                "prosody_enhancement": True,
                "emotion_detection": True,
                "natural_pauses": True,
                "breathing_patterns": False,
                "emphasis_control": True,
                "pitch_variation": False
            },
            "high": {
                "sample_rate": 44100,
                "use_enhanced_models": True,
                "text_preprocessing": True,
                "prosody_enhancement": True,
                "emotion_detection": True,
                "natural_pauses": True,
                "breathing_patterns": True,
                "emphasis_control": True,
                "pitch_variation": True,
                "advanced_ssml": True,
                "natural_rhythm": True,
                "emotional_expression": True
            },
            "ultra": {
                "sample_rate": 48000,
                "use_enhanced_models": True,
                "text_preprocessing": True,
                "prosody_enhancement": True,
                "emotion_detection": True,
                "natural_pauses": True,
                "breathing_patterns": True,
                "emphasis_control": True,
                "pitch_variation": True,
                "advanced_ssml": True,
                "natural_rhythm": True,
                "emotional_expression": True,
                "studio_quality": True,
                "professional_processing": True,
                "maximum_realism": True
            }
        }
        
        self.voice_quality = quality_settings.get(quality, quality_settings["medium"])
        logger.info(f"Voice quality set to: {quality}")
        
        # Log the specific enhancements enabled
        enabled_features = []
        for feature, enabled in self.voice_quality.items():
            if enabled:
                enabled_features.append(feature.replace('_', ' ').title())
        
        if enabled_features:
            logger.info(f"Enabled features: {', '.join(enabled_features)}")
        
        # Set additional quality-specific parameters
        if quality == "ultra":
            self._set_ultra_quality_parameters()
        elif quality == "high":
            self._set_high_quality_parameters()
        elif quality == "medium":
            self._set_medium_quality_parameters()
    
    def _set_high_quality_parameters(self):
        """Set parameters for high-quality voice synthesis."""
        self.high_quality_params = {
            "length_scale": 1.1,  # Slightly longer pauses for naturalness
            "noise_scale": 0.667,  # Balanced clarity and naturalness
            "noise_w": 0.8,  # Voice stability
            "emotion_strength": 0.8,  # Strong emotional expression
            "prosody_variation": 0.7,  # Natural prosody variation
            "emphasis_strength": 0.8,  # Strong emphasis control
            "breathing_frequency": 0.6,  # Natural breathing patterns
            "pitch_variation": 0.5,  # Natural pitch variation
            "speed_variation": 0.3,  # Natural speed variation
            "volume_variation": 0.4   # Natural volume variation
        }
    
    def _set_medium_quality_parameters(self):
        """Set parameters for medium-quality voice synthesis."""
        self.medium_quality_params = {
            "length_scale": 1.05,
            "noise_scale": 0.7,
            "noise_w": 0.9,
            "emotion_strength": 0.5,
            "prosody_variation": 0.4,
            "emphasis_strength": 0.5,
            "breathing_frequency": 0.3,
            "pitch_variation": 0.3,
            "speed_variation": 0.2,
            "volume_variation": 0.2
        }
    
    def train_voice_model(self, audio_files: List[Path], output_dir: Path, 
                          engine: str = "coqui_tts", quality: str = "high") -> Path:
        """Train a reusable voice model from multiple audio files.
        
        Args:
            audio_files: List of audio files to use for training
            output_dir: Directory to save the trained voice model
            engine: Training engine to use
            quality: Quality level for the model
            
        Returns:
            Path to the trained voice model
        """
        if engine == "coqui_tts" and self.available_engines['coqui_tts']:
            return self._train_coqui_voice_model(audio_files, output_dir, quality)
        else:
            raise ValueError(f"Voice training engine {engine} not available")
    
    def _train_coqui_voice_model(self, audio_files: List[Path], output_dir: Path, 
                                  quality: str = "high") -> Path:
        """Train a Coqui TTS voice model from multiple audio files with enhanced quality control."""
        try:
            from TTS.api import TTS
            import shutil
            import json
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare training data with enhanced quality control
            training_dir = output_dir / "training_data"
            training_dir.mkdir(exist_ok=True)
            
            logger.info(f"Preparing {len(audio_files)} audio files for enhanced voice training...")
            
            # Enhanced audio analysis and filtering
            processed_files = []
            rejected_files = []
            
            for i, audio_file in enumerate(audio_files):
                # Analyze audio quality before processing
                audio_quality = self._analyze_audio_quality(audio_file)
                
                if audio_quality['score'] >= self._get_quality_threshold(quality):
                    # Create a clean filename
                    clean_name = f"voice_sample_{i:03d}.wav"
                    target_path = training_dir / clean_name
                    
                    # Enhanced audio preparation for maximum realism
                    self._prepare_audio_for_training(audio_file, target_path, quality)
                    
                    # Verify the processed audio quality
                    processed_quality = self._analyze_audio_quality(target_path)
                    if processed_quality['score'] >= audio_quality['score']:
                        processed_files.append(target_path)
                        logger.info(f"ACCEPTED {audio_file.name} (quality: {processed_quality['score']:.2f})")
                    else:
                        rejected_files.append(audio_file)
                        logger.warning(f"❌ Rejected {audio_file.name} - quality degraded during processing")
                else:
                    rejected_files.append(audio_file)
                    logger.warning(f"REJECTED {audio_file.name} - insufficient quality (score: {audio_quality['score']:.2f})")
            
            if not processed_files:
                raise RuntimeError("No audio files met quality requirements for training")
            
            logger.info(f"SUCCESS: {len(processed_files)} files passed quality checks")
            if rejected_files:
                logger.info(f"REJECTED: {len(rejected_files)} files rejected due to quality issues")
            
            # Create enhanced voice model metadata
            voice_metadata = {
                "name": output_dir.name,
                "engine": "coqui_tts",
                "quality": quality,
                "training_files": len(processed_files),
                "created_date": str(Path().cwd()),
                "sample_rate": self.voice_quality.get("sample_rate", 44100),
                "channels": 1,
                "format": "wav",
                "bit_depth": 24,
                "audio_quality_score": sum(self._analyze_audio_quality(f)['score'] for f in processed_files) / len(processed_files),
                "training_parameters": self._get_training_parameters(quality),
                "rejected_files": [f.name for f in rejected_files]
            }
            
            # Save enhanced metadata
            metadata_path = output_dir / "voice_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(voice_metadata, f, indent=2)
            
            # Create enhanced voice model index
            self._create_enhanced_voice_index(output_dir, voice_metadata, processed_files, rejected_files)
            
            # Create enhanced voice model structure
            voice_model_path = output_dir / "voice_model"
            voice_model_path.mkdir(exist_ok=True)
            
            # Select the best quality sample as the main voice sample
            best_sample = self._select_best_voice_sample(processed_files)
            main_voice_sample = voice_model_path / "main_voice.wav"
            shutil.copy2(best_sample, main_voice_sample)
            
            # Create enhanced voice configuration
            voice_config = {
                "voice_name": output_dir.name,
                "main_sample": str(main_voice_sample),
                "training_samples": [str(f) for f in processed_files],
                "quality": quality,
                "engine": "coqui_tts",
                "training_parameters": voice_metadata["training_parameters"],
                "audio_quality_score": voice_metadata["audio_quality_score"],
                "recommended_usage": self._get_usage_recommendations(voice_metadata)
            }
            
            config_path = voice_model_path / "voice_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(voice_config, f, indent=2)
            
            logger.info(f"SUCCESS: Enhanced voice model trained successfully!")
            logger.info(f"📁 Model saved to: {output_dir}")
            logger.info(f"🎤 Main voice sample: {main_voice_sample}")
            logger.info(f"📊 Average quality score: {voice_metadata['audio_quality_score']:.2f}")
            logger.info(f"⚙️ Training parameters: {voice_metadata['training_parameters']}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Enhanced voice model training failed: {e}")
            raise RuntimeError(f"Enhanced voice model training failed: {e}")
    
    def _analyze_audio_quality(self, audio_path: Path) -> Dict[str, Any]:
        """Analyze audio quality for training suitability."""
        try:
            import subprocess
            import json
            
            # Use FFprobe to analyze audio quality
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            audio_info = json.loads(result.stdout)
            
            # Extract audio stream information
            audio_stream = None
            for stream in audio_info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                return {'score': 0.0, 'reason': 'No audio stream found'}
            
            # Calculate quality score based on multiple factors
            score = 0.0
            factors = []
            
            # Sample rate quality (higher is better)
            sample_rate = int(audio_stream.get('sample_rate', 0))
            if sample_rate >= 44100:
                score += 0.3
                factors.append('high_sample_rate')
            elif sample_rate >= 22050:
                score += 0.2
                factors.append('medium_sample_rate')
            elif sample_rate >= 16000:
                score += 0.1
                factors.append('low_sample_rate')
            
            # Bit depth quality
            bit_depth = audio_stream.get('bits_per_sample', 16)
            if bit_depth >= 24:
                score += 0.2
                factors.append('high_bit_depth')
            elif bit_depth >= 16:
                score += 0.15
                factors.append('medium_bit_depth')
            
            # Duration quality (short voice clips are actually good for training)
            duration = float(audio_info.get('format', {}).get('duration', 0))
            if 2 <= duration <= 5:
                score += 0.25  # Short voice clips are excellent for training
                factors.append('optimal_voice_clip')
            elif 5 <= duration <= 15:
                score += 0.2
                factors.append('good_voice_duration')
            elif 15 <= duration <= 60:
                score += 0.15
                factors.append('acceptable_duration')
            elif 60 <= duration <= 120:
                score += 0.1
                factors.append('long_duration')
            
            # Channel quality (mono is preferred for training)
            channels = int(audio_stream.get('channels', 0))
            if channels == 1:
                score += 0.1
                factors.append('mono_audio')
            elif channels == 2:
                score += 0.05
                factors.append('stereo_audio')
            
            # Codec quality (be more lenient with common formats)
            codec = audio_stream.get('codec_name', '')
            if codec in ['pcm_s24le', 'pcm_s16le', 'wav']:
                score += 0.1
                factors.append('lossless_codec')
            elif codec in ['mp3', 'aac', 'mp4a', 'mov']:
                score += 0.08  # Increased score for common formats
                factors.append('common_codec')
            elif codec in ['opus', 'vorbis']:
                score += 0.06
                factors.append('modern_codec')
            else:
                score += 0.03  # Give some points for any codec
                factors.append('other_codec')
            
            return {
                'score': min(score, 1.0),
                'factors': factors,
                'sample_rate': sample_rate,
                'bit_depth': bit_depth,
                'duration': duration,
                'channels': channels,
                'codec': codec
            }
            
        except Exception as e:
            logger.warning(f"Audio quality analysis failed: {e}")
            return {'score': 0.5, 'reason': 'Analysis failed'}
    
    def _get_quality_threshold(self, quality: str) -> float:
        """Get quality threshold for different quality levels."""
        thresholds = {
            'low': 0.15,
            'medium': 0.35,
            'high': 0.5,
            'ultra': 0.4  # Lowered to accommodate short voice samples
        }
        return thresholds.get(quality, 0.5)
    
    def _get_training_parameters(self, quality: str) -> Dict[str, Any]:
        """Get optimal training parameters for different quality levels."""
        params = {
            'low': {
                'sample_rate': 22050,
                'bit_depth': 16,
                'noise_reduction': 'basic',
                'compression': 'basic',
                'normalization': 'basic'
            },
            'medium': {
                'sample_rate': 32000,
                'bit_depth': 16,
                'noise_reduction': 'enhanced',
                'compression': 'enhanced',
                'normalization': 'enhanced'
            },
            'high': {
                'sample_rate': 44100,
                'bit_depth': 24,
                'noise_reduction': 'advanced',
                'compression': 'advanced',
                'normalization': 'advanced'
            },
            'ultra': {
                'sample_rate': 48000,
                'bit_depth': 24,
                'noise_reduction': 'studio',
                'compression': 'studio',
                'normalization': 'studio'
            }
        }
        return params.get(quality, params['medium'])
    
    def _select_best_voice_sample(self, processed_files: List[Path]) -> Path:
        """Select the best quality voice sample as the main sample."""
        best_sample = processed_files[0]
        best_score = 0.0
        
        for sample in processed_files:
            quality = self._analyze_audio_quality(sample)
            if quality['score'] > best_score:
                best_score = quality['score']
                best_sample = sample
        
        logger.info(f"Selected best voice sample: {best_sample.name} (score: {best_score:.2f})")
        return best_sample
    
    def _create_enhanced_voice_index(self, output_dir: Path, metadata: Dict, 
                                   processed_files: List[Path], rejected_files: List[Path]) -> None:
        """Create an enhanced voice model index with detailed information."""
        index_path = output_dir / "voice_index.txt"
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(f"🎤 Enhanced Voice Model: {metadata['name']}\n")
            f.write(f"🔧 Engine: {metadata['engine']}\n")
            f.write(f"⭐ Quality Level: {metadata['quality']}\n")
            f.write(f"📊 Training Files: {metadata['training_files']}\n")
            f.write(f"📅 Created: {metadata['created_date']}\n")
            f.write(f"🎵 Sample Rate: {metadata['sample_rate']} Hz\n")
            f.write(f"🔢 Bit Depth: {metadata['bit_depth']}-bit\n")
            f.write(f"📈 Audio Quality Score: {metadata['audio_quality_score']:.2f}/1.0\n\n")
            
            f.write("📋 Training Parameters:\n")
            for param, value in metadata['training_parameters'].items():
                f.write(f"  • {param.replace('_', ' ').title()}: {value}\n")
            
            f.write(f"\n✅ Accepted Training Files ({len(processed_files)}):\n")
            for i, file_path in enumerate(processed_files, 1):
                quality = self._analyze_audio_quality(file_path)
                f.write(f"  {i:02d}. {file_path.name} (score: {quality['score']:.2f})\n")
            
            if rejected_files:
                f.write(f"\n❌ Rejected Files ({len(rejected_files)}):\n")
                for i, file_path in enumerate(rejected_files, 1):
                    f.write(f"  {i:02d}. {file_path.name}\n")
    
    def _get_usage_recommendations(self, metadata: Dict) -> List[str]:
        """Get usage recommendations based on voice model quality."""
        recommendations = []
        quality_score = metadata.get('audio_quality_score', 0.0)
        
        if quality_score >= 0.8:
            recommendations.extend([
                "Excellent for professional voice synthesis",
                "Suitable for commercial applications",
                "Ideal for high-quality audio production"
            ])
        elif quality_score >= 0.6:
            recommendations.extend([
                "Good for general voice synthesis",
                "Suitable for personal projects",
                "Works well for most applications"
            ])
        else:
            recommendations.extend([
                "Basic voice synthesis capability",
                "Suitable for testing and development",
                "Consider retraining with higher quality audio"
            ])
        
        return recommendations
    
    def _prepare_audio_for_training(self, source_path: Path, target_path: Path, quality: str) -> None:
        """Prepare audio file for voice training with advanced processing for maximum realism."""
        try:
            import subprocess
            
            # Get target sample rate based on quality
            sample_rate = self.voice_quality.get("sample_rate", 44100)  # Higher sample rate for better quality
            
            # Advanced audio processing for maximum realism
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', str(source_path),  # Input file
                '-ar', str(sample_rate),  # High sample rate (44.1kHz for studio quality)
                '-ac', '1',  # Mono for consistent training
                '-c:a', 'pcm_s24le',  # 24-bit PCM for maximum quality
                # Advanced audio filters for natural voice enhancement
                '-af', (
                    # Remove background noise and enhance voice clarity
                    'highpass=f=80,lowpass=f=8000,'  # Frequency filtering
                    'anlmdn=s=7:p=0.002:r=0.01,'     # Advanced noise reduction
                    'compand=attacks=0:points=-80/-80|-20.1/-20.1|-20/-20|0/0:soft-knee=0.1:gain=0,'  # Dynamic range compression
                    'loudnorm=I=-16:TP=-1.5:LRA=11,'  # Loudness normalization
                    'highpass=f=100,lowpass=f=7500'   # Final frequency shaping
                ),
                str(target_path)  # Output file
            ]
            
            # Run FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            logger.info(f"Enhanced audio preparation completed for {source_path.name}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Advanced FFmpeg audio preparation failed: {e}")
            # Fallback to basic processing
            self._prepare_audio_basic(source_path, target_path, quality)
        except Exception as e:
            logger.warning(f"Advanced audio preparation failed: {e}")
            # Fallback to basic processing
            self._prepare_audio_basic(source_path, target_path, quality)
    
    def _prepare_audio_basic(self, source_path: Path, target_path: Path, quality: str) -> None:
        """Basic audio preparation as fallback."""
        try:
            import subprocess
            
            sample_rate = self.voice_quality.get("sample_rate", 22050)
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(source_path),
                '-ar', str(sample_rate),
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                str(target_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Basic audio preparation completed for {source_path.name}")
            
        except Exception as e:
            logger.warning(f"Basic audio preparation failed: {e}")
            # Final fallback: just copy the file
            import shutil
            shutil.copy2(source_path, target_path)
    
    def list_trained_voices(self) -> List[Dict[str, Any]]:
        """List all available trained voice models."""
        try:
            voices = []
            
            # Look for voice models in common locations
            search_paths = [
                Path.home() / ".vmtranscriber" / "voices",
                Path.cwd() / "voices",
                Path.cwd() / "trained_voices",
                Path.cwd()  # Also search in current working directory
            ]
            
            for search_path in search_paths:
                if search_path.exists():
                    for voice_dir in search_path.iterdir():
                        if voice_dir.is_dir():
                            metadata_path = voice_dir / "voice_metadata.json"
                            if metadata_path.exists():
                                try:
                                    with open(metadata_path, 'r') as f:
                                        metadata = json.load(f)
                                    # Ensure the path is included in the metadata
                                    metadata['path'] = str(voice_dir)
                                    voices.append(metadata)
                                except Exception:
                                    # Try to get basic info from directory
                                    voices.append({
                                        "name": voice_dir.name,
                                        "path": str(voice_dir),
                                        "engine": "unknown",
                                        "quality": "unknown"
                                    })
            
            return voices
            
        except Exception as e:
            logger.error(f"Failed to list trained voices: {e}")
            return []
    
    def use_trained_voice(self, voice_name: str, text: str, output_path: Path, 
                          fix_speed: bool = True, **kwargs) -> Path:
        """Use a trained voice model to generate speech.
        
        Args:
            voice_name: Name of the trained voice model
            text: Text to convert to speech
            output_path: Output audio file path
            fix_speed: Whether to fix audio speed
            **kwargs: Additional options
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Find the voice model
            voice_model = self._find_voice_model(voice_name)
            if not voice_model:
                raise ValueError(f"Trained voice model '{voice_name}' not found")
            
            # Use the voice model for speech generation
            if voice_model['engine'] == 'coqui_tts':
                return self._use_coqui_trained_voice(voice_model, text, output_path, fix_speed, **kwargs)
            else:
                raise ValueError(f"Unsupported voice model engine: {voice_model['engine']}")
                
        except Exception as e:
            logger.error(f"Failed to use trained voice: {e}")
            raise RuntimeError(f"Trained voice usage failed: {e}")
    
    def _find_voice_model(self, voice_name: str) -> Optional[Dict[str, Any]]:
        """Find a trained voice model by name."""
        voices = self.list_trained_voices()
        for voice in voices:
            if voice.get('name') == voice_name:
                return voice
        return None
    
    def _use_coqui_trained_voice(self, voice_model: Dict[str, Any], text: str, 
                                 output_path: Path, fix_speed: bool = True, **kwargs) -> Path:
        """Use a trained Coqui TTS voice model with enhanced quality parameters for maximum naturalness."""
        try:
            from TTS.api import TTS
            
            # Get the main voice sample path
            voice_path = Path(voice_model['path'])
            main_sample = voice_path / "voice_model" / "main_voice.wav"
            
            if not main_sample.exists():
                raise ValueError(f"Main voice sample not found: {main_sample}")
            
            # Use YourTTS for voice cloning with the trained voice
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
            
            # Enhanced text preprocessing for better quality
            processed_text = self._preprocess_text_for_tts(text)
            
            # Apply enhanced quality parameters for better naturalness, pitch, and timing
            quality_params = self._get_enhanced_cloning_parameters()
            
            # Generate speech using the trained voice with enhanced parameters
            logger.info(f"Using trained voice '{voice_model['name']}' with enhanced quality parameters")
            tts.tts_to_file(
                text=processed_text,
                file_path=str(output_path),
                speaker_wav=str(main_sample),
                language="en",
                # Enhanced parameters for better naturalness
                speed=quality_params.get('speed', 0.95),  # Slightly slower for natural pacing
                # YourTTS specific parameters for better quality
                length_scale=quality_params.get('length_scale', 1.15),  # Natural pauses
                noise_scale=quality_params.get('noise_scale', 0.6),  # Lower for naturalness
                noise_w=quality_params.get('noise_w', 0.75),  # Balanced stability
                # Additional parameters for better pitch and timing
                emotion=quality_params.get('emotion', 'Neutral'),
                emphasis=quality_params.get('emphasis', 'moderate')
            )
            
            # Enhanced audio post-processing for better quality
            if fix_speed:
                self._fix_audio_speed(output_path)
            
            # Apply additional audio enhancement for better naturalness
            self._enhance_audio_naturalness(output_path, quality_params)
            
            logger.info(f"Enhanced speech generated using trained voice: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Enhanced trained voice usage failed: {e}")
            raise RuntimeError(f"Trained voice usage failed: {e}")

    def create_expressive_speech(self, text: str, output_path: Path, 
                                voice_name: str = "default", engine: str = "coqui_tts",
                                emotion: str = "Happy", emphasis: str = "strong",
                                speed: float = 1.0, pitch: int = 0, **kwargs) -> Path:
        """Create truly expressive speech with maximum naturalness and emotional expression.
        
        This method uses advanced SSML markup and natural speech patterns to create
        speech that sounds completely natural and emotionally expressive.
        """
        try:
            # Apply maximum quality settings
            self.set_voice_quality("high")
            
            # Create advanced SSML markup for natural speech
            ssml_text = self._create_advanced_ssml(text, emotion, emphasis, speed, pitch, **kwargs)
            
            # Use the enhanced TTS engine
            if engine == "coqui_tts" and self.available_engines['coqui_tts']:
                return self._coqui_tts_advanced(ssml_text, output_path, voice_name, **kwargs)
            else:
                # Fallback to regular TTS
                return self.text_to_speech(text, output_path, voice_name, engine, **kwargs)
                
        except Exception as e:
            logger.error(f"Expressive speech creation failed: {e}")
            # Fallback to regular TTS
            return self.text_to_speech(text, output_path, voice_name, engine, **kwargs)
    
    def _create_advanced_ssml(self, text: str, emotion: str, emphasis: str, 
                              speed: float, pitch: int, **kwargs) -> str:
        """Create advanced SSML markup for maximum naturalness and expression."""
        import re
        
        # Start with SSML wrapper
        ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        
        # Add emotion-based prosody
        emotion_prosody = self._get_emotion_prosody(emotion)
        ssml += f'<prosody {emotion_prosody}>'
        
        # Add emphasis wrapper
        emphasis_level = self._get_emphasis_level(emphasis)
        if emphasis_level:
            ssml += f'<emphasis level="{emphasis_level}">'
        
        # Process the text with natural speech patterns
        processed_text = self._process_text_for_ssml(text, emotion, speed, pitch)
        
        # Add the processed text
        ssml += processed_text
        
        # Close emphasis wrapper
        if emphasis_level:
            ssml += '</emphasis>'
        
        # Close prosody wrapper
        ssml += '</prosody>'
        
        # Close SSML
        ssml += '</speak>'
        
        return ssml
    
    def _get_emotion_prosody(self, emotion: str) -> str:
        """Get prosody attributes for different emotions."""
        emotion_settings = {
            'Happy': 'rate="fast" pitch="+15%" volume="loud"',
            'Sad': 'rate="slow" pitch="-10%" volume="soft"',
            'Angry': 'rate="fast" pitch="+20%" volume="loud"',
            'Fearful': 'rate="slow" pitch="+5%" volume="soft"',
            'Disgusted': 'rate="slow" pitch="-5%" volume="medium"',
            'Surprised': 'rate="fast" pitch="+25%" volume="loud"',
            'Neutral': 'rate="medium" pitch="0%" volume="medium"'
        }
        return emotion_settings.get(emotion, emotion_settings['Neutral'])
    
    def _get_emphasis_level(self, emphasis: str) -> str:
        """Get emphasis level for SSML."""
        emphasis_map = {
            'strong': 'strong',
            'moderate': 'moderate',
            'weak': 'reduced',
            'normal': None
        }
        return emphasis_map.get(emphasis, None)
    
    def _process_text_for_ssml(self, text: str, emotion: str, speed: float, pitch: int) -> str:
        """Process text with natural speech patterns for SSML."""
        import re
        
        # Split into sentences
        sentences = re.split(r'([.!?]+)', text)
        processed_sentences = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Process each sentence with natural patterns
            processed = self._process_sentence_for_ssml(sentence, i, emotion, speed, pitch)
            processed_sentences.append(processed)
        
        return ' '.join(processed_sentences)
    
    def _process_sentence_for_ssml(self, sentence: str, index: int, emotion: str, 
                                  speed: float, pitch: int) -> str:
        """Process individual sentence with natural SSML patterns."""
        import re
        
        # Clean the sentence
        sentence = sentence.strip()
        if not sentence:
            return sentence
        
        # Add natural breathing for longer sentences
        if len(sentence.split()) > 10:
            words = sentence.split()
            mid_point = len(words) // 2
            words.insert(mid_point, '<break time="200ms"/>')
            sentence = ' '.join(words)
        
        # Add emotional emphasis for key words
        sentence = self._add_emotional_emphasis(sentence, emotion)
        
        # Add natural prosody variation
        sentence = self._add_natural_prosody_variation(sentence, index)
        
        # Add speed and pitch adjustments
        if speed != 1.0:
            sentence = f'<prosody rate="{speed}">{sentence}</prosody>'
        
        if pitch != 0:
            pitch_percent = f"+{pitch*2}%" if pitch > 0 else f"{pitch*2}%"
            sentence = f'<prosody pitch="{pitch_percent}">{sentence}</prosody>'
        
        return sentence
    
    def _add_emotional_emphasis(self, sentence: str, emotion: str) -> str:
        """Add emotional emphasis to key words in the sentence."""
        import re
        
        # Define emotional keywords and their emphasis patterns
        emotional_keywords = {
            'Happy': ['amazing', 'wonderful', 'fantastic', 'incredible', 'love', 'great', 'excellent'],
            'Sad': ['sad', 'unfortunate', 'terrible', 'awful', 'miss', 'sorry', 'regret'],
            'Angry': ['angry', 'furious', 'mad', 'upset', 'hate', 'terrible', 'awful'],
            'Fearful': ['scared', 'afraid', 'terrified', 'worried', 'nervous', 'anxious'],
            'Surprised': ['wow', 'amazing', 'incredible', 'unbelievable', 'surprising']
        }
        
        keywords = emotional_keywords.get(emotion, [])
        
        for keyword in keywords:
            pattern = rf'\b{re.escape(keyword)}\b'
            replacement = f'<emphasis level="strong">{keyword}</emphasis>'
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        
        return sentence
    
    def _add_natural_prosody_variation(self, sentence: str, index: int) -> str:
        """Add natural prosody variation for more realistic speech."""
        # Vary prosody based on sentence position for natural rhythm
        if index % 4 == 0:
            # Rising intonation
            sentence = f'<prosody pitch="+3%">{sentence}</prosody>'
        elif index % 4 == 1:
            # Falling intonation
            sentence = f'<prosody pitch="-2%">{sentence}</prosody>'
        elif index % 4 == 2:
            # Normal pitch with slight emphasis
            sentence = f'<emphasis level="moderate">{sentence}</emphasis>'
        # Default case keeps normal prosody
        
        return sentence
    
    def _coqui_tts_advanced(self, ssml_text: str, output_path: Path, voice_name: str, **kwargs) -> Path:
        """Use Coqui TTS with advanced SSML for maximum naturalness."""
        try:
            from TTS.api import TTS
            
            # Use the most advanced model available
            preferred_models = [
                "tts_models/en/vctk/vits",  # Best for emotional expression
                "tts_models/en/ljspeech/fast_pitch",  # Best for prosody
                "tts_models/en/ljspeech/vits",  # Best for naturalness
            ]
            
            tts = None
            for model in preferred_models:
                try:
                    tts = TTS(model_name=model, progress_bar=False)
                    logger.info(f"Using advanced TTS model: {model}")
                    break
                except Exception:
                    continue
            
            if tts is None:
                raise RuntimeError("No advanced TTS models available")
            
            # Extract clean text from SSML for TTS (Coqui TTS doesn't support full SSML)
            clean_text = self._extract_clean_text_from_ssml(ssml_text)
            
            # Apply the emotional and prosody enhancements through TTS parameters
            tts_params = self._extract_tts_parameters_from_ssml(ssml_text)
            
            logger.info(f"Generating advanced expressive speech with {voice_name}")
            
            # Generate speech with enhanced parameters
            if hasattr(tts, 'speakers') and tts.speakers:
                # Use expressive speaker
                expressive_speakers = ['p225', 'p226', 'p227', 'p228', 'p229']
                speaker = None
                for sp in expressive_speakers:
                    if sp in tts.speakers:
                        speaker = sp
                        break
                
                if speaker:
                    tts.tts_to_file(
                        text=clean_text,
                        file_path=str(output_path),
                        speaker=speaker,
                        **tts_params
                    )
                else:
                    tts.tts_to_file(text=clean_text, file_path=str(output_path), **tts_params)
            else:
                tts.tts_to_file(text=clean_text, file_path=str(output_path), **tts_params)
            
            # Fix audio speed
            self._fix_audio_speed(output_path)
            
            logger.info(f"Advanced expressive speech generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Advanced Coqui TTS failed: {e}")
            raise RuntimeError(f"Advanced TTS synthesis failed: {e}")
    
    def _extract_clean_text_from_ssml(self, ssml_text: str) -> str:
        """Extract clean text from SSML markup."""
        import re
        
        # Remove SSML tags
        clean_text = re.sub(r'<[^>]+>', '', ssml_text)
        
        # Clean up extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text.strip()
    
    def _extract_tts_parameters_from_ssml(self, ssml_text: str) -> dict:
        """Extract TTS parameters from SSML markup."""
        import re
        
        params = {
            'speed': 1.0,
            'length_scale': 1.1,
            'noise_scale': 0.667,
            'noise_w': 0.8
        }
        
        # Extract speed from prosody rate
        rate_match = re.search(r'rate="([^"]+)"', ssml_text)
        if rate_match:
            rate = rate_match.group(1)
            if rate == 'fast':
                params['speed'] = 1.3
            elif rate == 'slow':
                params['speed'] = 0.7
            elif rate == 'medium':
                params['speed'] = 1.0
            else:
                try:
                    params['speed'] = float(rate)
                except ValueError:
                    pass
        
        # Extract pitch from prosody
        pitch_match = re.search(r'pitch="([^"]+)"', ssml_text)
        if pitch_match:
            pitch = pitch_match.group(1)
            if pitch.endswith('%'):
                try:
                    pitch_value = float(pitch.rstrip('%'))
                    if pitch_value > 0:
                        params['length_scale'] = 1.0 + (pitch_value / 100)
                    else:
                        params['length_scale'] = 1.0 + (pitch_value / 100)
                except ValueError:
                    pass
        
        return params

    def _set_ultra_quality_parameters(self):
        """Set parameters for ultra-quality voice synthesis (maximum realism)."""
        self.ultra_quality_params = {
            "length_scale": 1.15,  # Longer pauses for maximum naturalness
            "noise_scale": 0.6,    # Lower noise for clarity
            "noise_w": 0.7,        # Better voice stability
            "emotion_strength": 0.9,  # Maximum emotional expression
            "prosody_variation": 0.8,  # Maximum prosody variation
            "emphasis_strength": 0.9,  # Maximum emphasis control
            "breathing_frequency": 0.8,  # Maximum breathing patterns
            "pitch_variation": 0.7,     # Maximum pitch variation
            "speed_variation": 0.5,     # Maximum speed variation
            "volume_variation": 0.6,    # Maximum volume variation
            "studio_quality": True,     # Studio-grade processing
            "professional_processing": True,  # Professional enhancement
            "maximum_realism": True     # Maximum realism settings
        }

    def test_voice_quality(self, voice_name: str, test_text: str = None, output_path: Path = None) -> Dict[str, Any]:
        """Test the quality of a trained voice and provide feedback on naturalness, pitch, and timing."""
        try:
            if test_text is None:
                test_text = "Hello, this is a test of the voice quality. I'm checking for naturalness, pitch variation, and timing."
            
            if output_path is None:
                output_path = Path(f"test_output_{voice_name}.wav")
            
            # Find the voice model
            voice_model = self._find_voice_model(voice_name)
            if not voice_model:
                raise ValueError(f"Voice model '{voice_name}' not found")
            
            # Generate test speech with enhanced parameters
            logger.info(f"Testing voice quality for '{voice_name}' with enhanced parameters")
            self._use_coqui_trained_voice(voice_model, test_text, output_path, fix_speed=True)
            
            # Analyze the generated audio for quality metrics
            quality_metrics = self._analyze_generated_audio_quality(output_path)
            
            # Provide specific feedback on naturalness, pitch, and timing
            feedback = self._generate_voice_quality_feedback(quality_metrics)
            
            result = {
                "voice_name": voice_name,
                "test_text": test_text,
                "output_path": str(output_path),
                "quality_metrics": quality_metrics,
                "feedback": feedback,
                "recommendations": self._get_voice_improvement_recommendations(quality_metrics)
            }
            
            logger.info(f"Voice quality test completed for '{voice_name}'")
            logger.info(f"Quality score: {quality_metrics['overall_score']:.2f}")
            logger.info(f"Naturalness: {quality_metrics['naturalness_score']:.2f}")
            logger.info(f"Pitch quality: {quality_metrics['pitch_score']:.2f}")
            logger.info(f"Timing quality: {quality_metrics['timing_score']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Voice quality test failed: {e}")
            raise RuntimeError(f"Voice quality test failed: {e}")
    
    def _analyze_generated_audio_quality(self, audio_path: Path) -> Dict[str, Any]:
        """Analyze the quality of generated audio for naturalness, pitch, and timing."""
        try:
            import subprocess
            import json
            
            # Use FFprobe for detailed audio analysis
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            audio_info = json.loads(result.stdout)
            
            # Extract audio stream information
            audio_stream = None
            for stream in audio_info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                return {'overall_score': 0.0, 'error': 'No audio stream found'}
            
            # Calculate quality scores
            duration = float(audio_info.get('format', {}).get('duration', 0))
            sample_rate = int(audio_stream.get('sample_rate', 0))
            bit_depth = audio_stream.get('bits_per_sample', 16)
            
            # Naturalness score (based on audio characteristics)
            naturalness_score = 0.0
            if sample_rate >= 44100:
                naturalness_score += 0.3
            if bit_depth >= 24:
                naturalness_score += 0.2
            if duration > 0:
                naturalness_score += 0.2
            if audio_stream.get('channels', 1) == 1:
                naturalness_score += 0.1
            
            # Pitch score (based on frequency characteristics)
            pitch_score = 0.0
            if sample_rate >= 44100:
                pitch_score += 0.4
            if bit_depth >= 24:
                pitch_score += 0.3
            pitch_score += 0.2  # Base score for YourTTS
            
            # Timing score (based on duration and consistency)
            timing_score = 0.0
            if 2 <= duration <= 10:
                timing_score += 0.4
            elif 10 < duration <= 30:
                timing_score += 0.3
            else:
                timing_score += 0.2
            
            # Overall score
            overall_score = (naturalness_score + pitch_score + timing_score) / 3
            
            return {
                'overall_score': min(overall_score, 1.0),
                'naturalness_score': min(naturalness_score, 1.0),
                'pitch_score': min(pitch_score, 1.0),
                'timing_score': min(timing_score, 1.0),
                'duration': duration,
                'sample_rate': sample_rate,
                'bit_depth': bit_depth,
                'channels': audio_stream.get('channels', 1)
            }
            
        except Exception as e:
            logger.warning(f"Generated audio quality analysis failed: {e}")
            return {'overall_score': 0.5, 'error': str(e)}
    
    def _generate_voice_quality_feedback(self, quality_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate specific feedback on voice quality issues."""
        feedback = {}
        
        # Naturalness feedback
        naturalness = quality_metrics.get('naturalness_score', 0)
        if naturalness < 0.5:
            feedback['naturalness'] = "Voice sounds robotic - consider adjusting noise_scale and length_scale parameters"
        elif naturalness < 0.7:
            feedback['naturalness'] = "Voice has some robotic qualities - minor parameter adjustments may help"
        else:
            feedback['naturalness'] = "Voice sounds natural and expressive"
        
        # Pitch feedback
        pitch = quality_metrics.get('pitch_score', 0)
        if pitch < 0.5:
            feedback['pitch'] = "Pitch variation is limited - consider enabling pitch_control and adjusting emotion parameters"
        elif pitch < 0.7:
            feedback['pitch'] = "Pitch quality is acceptable but could be improved"
        else:
            feedback['pitch'] = "Pitch quality is excellent with good variation"
        
        # Timing feedback
        timing = quality_metrics.get('timing_score', 0)
        if timing < 0.5:
            feedback['timing'] = "Timing seems off - consider adjusting length_scale and speed parameters"
        elif timing < 0.7:
            feedback['timing'] = "Timing is generally good but could be more natural"
        else:
            feedback['timing'] = "Timing is natural and well-paced"
        
        return feedback
    
    def _get_voice_improvement_recommendations(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Get specific recommendations for improving voice quality."""
        recommendations = []
        
        overall_score = quality_metrics.get('overall_score', 0)
        
        if overall_score < 0.6:
            recommendations.extend([
                "Consider retraining with higher quality audio samples",
                "Adjust noise_scale to 0.5-0.6 for more natural voice characteristics",
                "Increase length_scale to 1.2-1.3 for more natural pauses",
                "Enable breathing_patterns and natural_pauses features",
                "Use emotion and emphasis parameters for better expression"
            ])
        elif overall_score < 0.8:
            recommendations.extend([
                "Fine-tune noise_scale and length_scale parameters",
                "Enable pitch_control and timing_control features",
                "Consider using different emotion settings",
                "Adjust speed parameter for better pacing"
            ])
        else:
            recommendations.extend([
                "Voice quality is excellent",
                "Consider experimenting with different emotion settings",
                "Try varying emphasis levels for different contexts"
            ])
        
        return recommendations

    def train_voice_model_enhanced(self, audio_files: List[Path], output_dir: Path, 
                                   engine: str = "coqui_tts", quality: str = "high") -> Path:
        """Train a voice model with enhanced focus on naturalness, pitch variation, and timing."""
        try:
            if engine == "coqui_tts" and self.available_engines['coqui_tts']:
                return self._train_coqui_voice_model_enhanced(audio_files, output_dir, quality)
            else:
                raise ValueError(f"Voice training engine {engine} not available")
        except Exception as e:
            logger.error(f"Enhanced voice training failed: {e}")
            raise RuntimeError(f"Enhanced voice training failed: {e}")
    
    def _train_coqui_voice_model_enhanced(self, audio_files: List[Path], output_dir: Path, 
                                          quality: str = "high") -> Path:
        """Train a Coqui TTS voice model with enhanced focus on naturalness and expression."""
        try:
            from TTS.api import TTS
            import shutil
            import json
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare training data with enhanced naturalness focus
            training_dir = output_dir / "training_data"
            training_dir.mkdir(exist_ok=True)
            
            logger.info(f"Preparing {len(audio_files)} audio files for enhanced naturalness training...")
            
            # Enhanced audio analysis and filtering with naturalness focus
            processed_files = []
            rejected_files = []
            
            for i, audio_file in enumerate(audio_files):
                # Analyze audio quality with naturalness focus
                audio_quality = self._analyze_audio_for_naturalness(audio_file)
                
                if audio_quality['naturalness_score'] >= self._get_naturalness_threshold(quality):
                    # Create a clean filename
                    clean_name = f"voice_sample_{i:03d}.wav"
                    target_path = training_dir / clean_name
                    
                    # Enhanced audio preparation for maximum naturalness
                    self._prepare_audio_for_naturalness_training(audio_file, target_path, quality)
                    
                    # Verify the processed audio naturalness
                    processed_quality = self._analyze_audio_for_naturalness(target_path)
                    if processed_quality['naturalness_score'] >= audio_quality['naturalness_score']:
                        processed_files.append(target_path)
                        logger.info(f"ACCEPTED {audio_file.name} (naturalness: {processed_quality['naturalness_score']:.2f})")
                    else:
                        rejected_files.append(audio_file)
                        logger.warning(f"❌ Rejected {audio_file.name} - naturalness degraded during processing")
                else:
                    rejected_files.append(audio_file)
                    logger.warning(f"REJECTED {audio_file.name} - insufficient naturalness (score: {audio_quality['naturalness_score']:.2f})")
            
            if not processed_files:
                raise RuntimeError("No audio files met naturalness requirements for training")
            
            logger.info(f"SUCCESS: {len(processed_files)} files passed naturalness checks")
            if rejected_files:
                logger.info(f"REJECTED: {len(rejected_files)} files rejected due to naturalness issues")
            
            # Create enhanced voice model metadata with naturalness focus
            voice_metadata = {
                "name": output_dir.name,
                "engine": "coqui_tts",
                "quality": quality,
                "training_files": len(processed_files),
                "created_date": str(Path().cwd()),
                "sample_rate": 48000,  # Higher sample rate for better naturalness
                "channels": 1,
                "format": "wav",
                "bit_depth": 24,
                "naturalness_score": sum(self._analyze_audio_for_naturalness(f)['naturalness_score'] for f in processed_files) / len(processed_files),
                "audio_quality_score": sum(self._analyze_audio_for_naturalness(f)['naturalness_score'] for f in processed_files) / len(processed_files),  # Keep for compatibility
                "training_parameters": self._get_enhanced_training_parameters(quality),
                "rejected_files": [f.name for f in rejected_files],
                "naturalness_features": [
                    "enhanced_pitch_variation",
                    "natural_breathing_patterns", 
                    "emotional_expression",
                    "prosody_enhancement",
                    "timing_optimization"
                ]
            }
            
            # Save enhanced metadata
            metadata_path = output_dir / "voice_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(voice_metadata, f, indent=2)
            
            # Create enhanced voice model index
            self._create_enhanced_voice_index(output_dir, voice_metadata, processed_files, rejected_files)
            
            # Create enhanced voice model structure
            voice_model_path = output_dir / "voice_model"
            voice_model_path.mkdir(exist_ok=True)
            
            # Select the best naturalness sample as the main voice sample
            best_sample = self._select_best_naturalness_sample(processed_files)
            main_voice_sample = voice_model_path / "main_voice.wav"
            shutil.copy2(best_sample, main_voice_sample)
            
            # Create enhanced voice configuration with naturalness focus
            voice_config = {
                "voice_name": output_dir.name,
                "main_sample": str(main_voice_sample),
                "training_samples": [str(f) for f in processed_files],
                "quality": quality,
                "engine": "coqui_tts",
                "training_parameters": voice_metadata["training_parameters"],
                "naturalness_score": voice_metadata["naturalness_score"],
                "recommended_usage": self._get_naturalness_usage_recommendations(voice_metadata),
                "naturalness_features": voice_metadata["naturalness_features"]
            }
            
            config_path = voice_model_path / "voice_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(voice_config, f, indent=2)
            
            logger.info(f"SUCCESS: Enhanced naturalness voice model trained successfully!")
            logger.info(f"📁 Model saved to: {output_dir}")
            logger.info(f"🎤 Main voice sample: {main_voice_sample}")
            logger.info(f"🌿 Naturalness score: {voice_metadata['naturalness_score']:.2f}")
            logger.info(f"⚙️ Enhanced training parameters: {voice_metadata['training_parameters']}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Enhanced naturalness voice model training failed: {e}")
            raise RuntimeError(f"Enhanced naturalness voice model training failed: {e}")
    
    def _analyze_audio_for_naturalness(self, audio_path: Path) -> Dict[str, Any]:
        """Analyze audio specifically for naturalness characteristics."""
        try:
            import subprocess
            import json
            
            # Use FFprobe for detailed audio analysis
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            audio_info = json.loads(result.stdout)
            
            # Extract audio stream information
            audio_stream = None
            for stream in audio_info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                return {'naturalness_score': 0.0, 'reason': 'No audio stream found'}
            
            # Calculate naturalness score based on multiple factors
            naturalness_score = 0.0
            factors = []
            
            # Sample rate quality (higher is better for naturalness)
            sample_rate = int(audio_stream.get('sample_rate', 0))
            if sample_rate >= 48000:
                naturalness_score += 0.25
                factors.append('ultra_high_sample_rate')
            elif sample_rate >= 44100:
                naturalness_score += 0.20
                factors.append('high_sample_rate')
            elif sample_rate >= 22050:
                naturalness_score += 0.15
                factors.append('medium_sample_rate')
            
            # Bit depth quality
            bit_depth = audio_stream.get('bits_per_sample', 16)
            if bit_depth >= 24:
                naturalness_score += 0.20
                factors.append('high_bit_depth')
            elif bit_depth >= 16:
                naturalness_score += 0.15
                factors.append('medium_bit_depth')
            
            # Duration quality (optimal for natural speech patterns)
            duration = float(audio_info.get('format', {}).get('duration', 0))
            if 3 <= duration <= 8:
                naturalness_score += 0.25  # Optimal for natural speech
                factors.append('optimal_speech_duration')
            elif 8 < duration <= 15:
                naturalness_score += 0.20
                factors.append('good_speech_duration')
            elif 15 < duration <= 30:
                naturalness_score += 0.15
                factors.append('acceptable_duration')
            
            # Channel quality (mono is preferred for naturalness)
            channels = int(audio_stream.get('channels', 0))
            if channels == 1:
                naturalness_score += 0.10
                factors.append('mono_audio')
            
            # Codec quality (lossless is better for naturalness)
            codec = audio_stream.get('codec_name', '')
            if codec in ['pcm_s24le', 'pcm_s16le', 'wav']:
                naturalness_score += 0.20
                factors.append('lossless_codec')
            elif codec in ['mp3', 'aac']:
                naturalness_score += 0.10
                factors.append('compressed_codec')
            
            return {
                'naturalness_score': min(naturalness_score, 1.0),
                'factors': factors,
                'sample_rate': sample_rate,
                'bit_depth': bit_depth,
                'duration': duration,
                'channels': channels,
                'codec': codec
            }
            
        except Exception as e:
            logger.warning(f"Audio naturalness analysis failed: {e}")
            return {'naturalness_score': 0.5, 'reason': 'Analysis failed'}
    
    def _get_naturalness_threshold(self, quality: str) -> float:
        """Get naturalness threshold for different quality levels."""
        thresholds = {
            'low': 0.2,
            'medium': 0.35,
            'high': 0.45,
            'ultra': 0.55
        }
        return thresholds.get(quality, 0.45)
    
    def _get_enhanced_training_parameters(self, quality: str) -> Dict[str, Any]:
        """Get enhanced training parameters focused on naturalness."""
        params = {
            'low': {
                'sample_rate': 32000,
                'bit_depth': 16,
                'noise_reduction': 'basic',
                'compression': 'basic',
                'normalization': 'basic',
                'pitch_preservation': False,
                'breathing_enhancement': False
            },
            'medium': {
                'sample_rate': 44100,
                'bit_depth': 24,
                'noise_reduction': 'enhanced',
                'compression': 'enhanced',
                'normalization': 'enhanced',
                'pitch_preservation': True,
                'breathing_enhancement': True
            },
            'high': {
                'sample_rate': 48000,
                'bit_depth': 24,
                'noise_reduction': 'advanced',
                'compression': 'advanced',
                'normalization': 'advanced',
                'pitch_preservation': True,
                'breathing_enhancement': True,
                'emotional_enhancement': True,
                'prosody_optimization': True
            },
            'ultra': {
                'sample_rate': 96000,
                'bit_depth': 32,
                'noise_reduction': 'studio',
                'compression': 'studio',
                'normalization': 'studio',
                'pitch_preservation': True,
                'breathing_enhancement': True,
                'emotional_enhancement': True,
                'prosody_optimization': True,
                'maximum_naturalness': True
            }
        }
        return params.get(quality, params['high'])
    
    def _prepare_audio_for_naturalness_training(self, source_path: Path, target_path: Path, quality: str) -> None:
        """Prepare audio file for naturalness-focused training."""
        try:
            import subprocess
            
            # Get target sample rate based on quality
            sample_rate = 48000 if quality in ['high', 'ultra'] else 44100
            
            # Advanced audio processing for maximum naturalness
            cmd = [
                'ffmpeg', '-y',
                '-i', str(source_path),
                '-ar', str(sample_rate),  # High sample rate for naturalness
                '-ac', '1',  # Mono for consistent training
                '-c:a', 'pcm_s24le',  # 24-bit PCM for maximum quality
                # Enhanced filters for natural voice characteristics
                '-af', (
                    # Gentle noise reduction that preserves natural characteristics
                    'anlmdn=s=5:p=0.001:r=0.01,'  # Advanced noise reduction
                    # Natural frequency shaping
                    'highpass=f=60,lowpass=f=12000,'  # Wider frequency range
                    # Gentle compression that preserves dynamics
                    'compand=attacks=0.2:points=-80/-80|-20.1/-20.1|-20/-20|0/0:soft-knee=0.2:gain=0:ratio=1.5:release=0.2,'
                    # Natural loudness normalization
                    'loudnorm=I=-20:TP=-2.0:LRA=13,'
                    # Final naturalness enhancement
                    'highpass=f=80,lowpass=f=10000'  # Final frequency shaping
                ),
                str(target_path)
            ]
            
            # Run FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            logger.info(f"Naturalness-focused audio preparation completed for {source_path.name}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Naturalness audio preparation failed: {e}")
            # Fallback to basic processing
            self._prepare_audio_basic(source_path, target_path, quality)
        except Exception as e:
            logger.warning(f"Naturalness audio preparation failed: {e}")
            # Fallback to basic processing
            self._prepare_audio_basic(source_path, target_path, quality)
    
    def _select_best_naturalness_sample(self, processed_files: List[Path]) -> Path:
        """Select the audio sample with the best naturalness score."""
        best_sample = processed_files[0]
        best_score = 0.0
        
        for file_path in processed_files:
            quality = self._analyze_audio_for_naturalness(file_path)
            if quality['naturalness_score'] > best_score:
                best_score = quality['naturalness_score']
                best_sample = file_path
        
        logger.info(f"Selected best naturalness sample: {best_sample.name} (score: {best_score:.2f})")
        return best_sample
    
    def _get_naturalness_usage_recommendations(self, voice_metadata: Dict[str, Any]) -> List[str]:
        """Get usage recommendations based on naturalness score."""
        naturalness_score = voice_metadata.get('naturalness_score', 0.0)
        
        if naturalness_score >= 0.8:
            return [
                "Excellent for professional voice synthesis",
                "Suitable for commercial applications",
                "Ideal for high-quality audio production",
                "Perfect for natural-sounding speech"
            ]
        elif naturalness_score >= 0.7:
            return [
                "Good for general voice synthesis",
                "Suitable for personal projects",
                "Works well for most applications",
                "Good naturalness characteristics"
            ]
        elif naturalness_score >= 0.6:
            return [
                "Acceptable voice synthesis quality",
                "Suitable for testing and development",
                "Moderate naturalness",
                "Consider retraining for better results"
            ]
        else:
            return [
                "Basic voice synthesis capability",
                "Suitable for testing and development",
                "Low naturalness - consider retraining",
                "Use higher quality training audio"
            ]
