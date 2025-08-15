"""
Output formatting and file writing for VMTranscriber.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import timedelta
import json

logger = logging.getLogger(__name__)

class OutputFormatter:
    """Handles output formatting and file writing."""
    
    def __init__(self, config):
        """Initialize output formatter with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT/VTT timestamp.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        ms = int(seconds * 1000)
        td = timedelta(milliseconds=ms)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = ms % 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    
    def write_plain_text(self, segments: List[Dict[str, Any]], output_path: Path):
        """Write plain text transcript.
        
        Args:
            segments: List of transcription segments
            output_path: Output file path
        """
        try:
            with output_path.open("w", encoding="utf-8") as f:
                for segment in segments:
                    f.write(segment["text"].strip() + " ")
            logger.info(f"Plain text transcript written to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write plain text transcript: {e}")
            raise
    
    def write_srt(self, segments: List[Dict[str, Any]], output_path: Path):
        """Write SRT subtitle file.
        
        Args:
            segments: List of transcription segments
            output_path: Output file path
        """
        try:
            with output_path.open("w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{self.format_timestamp(segment['start'])} --> {self.format_timestamp(segment['end'])}\n")
                    f.write(segment["text"].strip() + "\n\n")
            logger.info(f"SRT subtitles written to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write SRT subtitles: {e}")
            raise
    
    def write_vtt(self, segments: List[Dict[str, Any]], output_path: Path):
        """Write VTT caption file.
        
        Args:
            segments: List of transcription segments
            output_path: Output file path
        """
        try:
            with output_path.open("w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for segment in segments:
                    start = self.format_timestamp(segment['start']).replace(",", ".")
                    end = self.format_timestamp(segment['end']).replace(",", ".")
                    f.write(f"{start} --> {end}\n")
                    f.write(segment["text"].strip() + "\n\n")
            logger.info(f"VTT captions written to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write VTT captions: {e}")
            raise
    
    def write_json(self, segments: List[Dict[str, Any]], output_path: Path, metadata: Optional[Dict] = None):
        """Write JSON transcript with metadata.
        
        Args:
            segments: List of transcription segments
            output_path: Output file path
            metadata: Additional metadata to include
        """
        try:
            output_data = {
                "metadata": metadata or {},
                "segments": segments,
                "summary": {
                    "total_segments": len(segments),
                    "total_duration": segments[-1]["end"] if segments else 0,
                    "language": metadata.get("language", "unknown") if metadata else "unknown"
                }
            }
            
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON transcript written to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write JSON transcript: {e}")
            raise
    
    def write_csv(self, segments: List[Dict[str, Any]], output_path: Path):
        """Write CSV transcript with timestamps.
        
        Args:
            segments: List of transcription segments
            output_path: Output file path
        """
        try:
            import csv
            
            with output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Start Time", "End Time", "Duration", "Text"])
                
                for segment in segments:
                    duration = segment["end"] - segment["start"]
                    writer.writerow([
                        self.format_timestamp(segment["start"]),
                        self.format_timestamp(segment["end"]),
                        f"{duration:.2f}s",
                        segment["text"].strip()
                    ])
            logger.info(f"CSV transcript written to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write CSV transcript: {e}")
            raise
    
    def write_markdown(self, segments: List[Dict[str, Any]], output_path: Path, metadata: Optional[Dict] = None):
        """Write Markdown transcript with timestamps.
        
        Args:
            segments: List of transcription segments
            output_path: Output file path
            metadata: Additional metadata to include
        """
        try:
            with output_path.open("w", encoding="utf-8") as f:
                # Write header
                f.write("# Audio Transcript\n\n")
                
                if metadata:
                    f.write("## Metadata\n\n")
                    for key, value in metadata.items():
                        f.write(f"- **{key.title()}**: {value}\n")
                    f.write("\n")
                
                f.write("## Transcript\n\n")
                
                # Write segments with timestamps
                for i, segment in enumerate(segments, 1):
                    start_time = self.format_timestamp(segment["start"])
                    end_time = self.format_timestamp(segment["end"])
                    f.write(f"**{start_time} - {end_time}**\n")
                    f.write(f"{segment['text'].strip()}\n\n")
            logger.info(f"Markdown transcript written to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write Markdown transcript: {e}")
            raise
    
    def write_all_formats(self, segments: List[Dict[str, Any]], base_path: Path, 
                          formats: Optional[List[str]] = None, metadata: Optional[Dict] = None):
        """Write transcript in multiple formats.
        
        Args:
            segments: List of transcription segments
            base_path: Base path for output files (without extension)
            formats: List of output formats. If None, uses config default
            metadata: Additional metadata for formats that support it
            
        Returns:
            List of created output file paths
        """
        if formats is None:
            formats = self.config.get('output_formats', ['txt', 'srt', 'vtt'])
        
        output_files = []
        
        for format_type in formats:
            try:
                if format_type == 'txt':
                    output_path = base_path.with_name(f"{base_path.name}.txt")
                    self.write_plain_text(segments, output_path)
                    output_files.append(output_path)
                
                elif format_type == 'srt':
                    output_path = base_path.with_name(f"{base_path.name}.srt")
                    self.write_srt(segments, output_path)
                    output_files.append(output_path)
                
                elif format_type == 'vtt':
                    output_path = base_path.with_name(f"{base_path.name}.vtt")
                    self.write_vtt(segments, output_path)
                    output_files.append(output_path)
                
                elif format_type == 'json':
                    output_path = base_path.with_name(f"{base_path.name}.json")
                    self.write_json(segments, output_path, metadata)
                    output_files.append(output_path)
                
                elif format_type == 'csv':
                    output_path = base_path.with_name(f"{base_path.name}.csv")
                    self.write_csv(segments, output_path)
                    output_files.append(output_path)
                
                elif format_type == 'md':
                    output_path = base_path.with_name(f"{base_path.name}.md")
                    self.write_markdown(segments, output_path, metadata)
                    output_files.append(output_path)
                
                else:
                    logger.warning(f"Unknown output format: {format_type}")
                    
            except Exception as e:
                logger.error(f"Failed to write {format_type} format: {e}")
                # Continue with other formats
        
        return output_files
