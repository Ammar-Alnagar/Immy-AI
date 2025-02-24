import edge_tts
import io
import base64
from collections import deque
import asyncio
from typing import Dict, Optional, Set
import time
from datetime import datetime
from .logger import Logger

class TTSService:
    def __init__(self):
        self.logger = Logger("TTSService")
        self.voice = "en-US-JennyNeural"
        self.rate = "+0%"
        self.pitch = "+0Hz"
        self.tts_queue = deque()
        self.processing = False
        self.active_chunks: Set[int] = set()
        self.logger.info("TTSService initialized successfully", {
            "voice": self.voice,
            "rate": self.rate,
            "pitch": self.pitch
        })

    async def text_to_speech(self, text: str, chunk_id: int, sequence: int, metrics: Dict[str, int]) -> Optional[str]:
        """Convert text to speech and return base64 encoded audio data."""
        if not text.strip():
            self.logger.debug("Empty text received, skipping TTS")
            return None

        start_time = datetime.now().timestamp()
        try:
            self.logger.debug("Starting TTS conversion", {
                "chunk_id": chunk_id,
                "sequence": sequence,
                "text_length": len(text)
            })
            metrics['ttsStart'] = round(time.perf_counter() * 1000)
            
            # Create WAV header
            wav_header = io.BytesIO()
            # WAV header format:
            # RIFF header
            wav_header.write(b'RIFF')  # ChunkID
            wav_header.write(b'\x00\x00\x00\x00')  # ChunkSize (placeholder)
            wav_header.write(b'WAVE')  # Format
            
            # fmt subchunk
            wav_header.write(b'fmt ')  # Subchunk1ID
            wav_header.write((16).to_bytes(4, 'little'))  # Subchunk1Size (16 for PCM)
            wav_header.write((1).to_bytes(2, 'little'))  # AudioFormat (1 for PCM)
            wav_header.write((1).to_bytes(2, 'little'))  # NumChannels (1 for mono)
            wav_header.write((24000).to_bytes(4, 'little'))  # SampleRate (24kHz for Edge TTS)
            wav_header.write((48000).to_bytes(4, 'little'))  # ByteRate (SampleRate * NumChannels * BitsPerSample/8)
            wav_header.write((2).to_bytes(2, 'little'))  # BlockAlign (NumChannels * BitsPerSample/8)
            wav_header.write((16).to_bytes(2, 'little'))  # BitsPerSample (16 bits)
            
            # data subchunk
            wav_header.write(b'data')  # Subchunk2ID
            wav_header.write(b'\x00\x00\x00\x00')  # Subchunk2Size (placeholder)
            
            # Get audio data from Edge TTS
            audio_data = io.BytesIO()
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])
            
            audio_size = len(audio_data.getvalue())
            if not audio_size:
                self.logger.warning("No audio data generated", {
                    "chunk_id": chunk_id,
                    "text": text
                })
                return None
            
            # Update WAV header with correct sizes
            wav_header.seek(4)
            wav_header.write((36 + audio_size).to_bytes(4, 'little'))  # ChunkSize
            wav_header.seek(40)
            wav_header.write(audio_size.to_bytes(4, 'little'))  # Subchunk2Size
            
            # Combine header and audio data
            wav_data = io.BytesIO()
            wav_data.write(wav_header.getvalue())
            wav_data.write(audio_data.getvalue())
            
            # Get the WAV data and encode to base64
            wav_data.seek(0)
            audio_base64 = base64.b64encode(wav_data.getvalue()).decode('utf-8')
            
            metrics['ttsEnd'] = round(time.perf_counter() * 1000)
            duration_ms = round((datetime.now().timestamp() - start_time) * 1000, 2)
            
            self.logger.api_call(
                "Edge TTS",
                success=True,
                duration_ms=duration_ms,
                context={
                    "chunk_id": chunk_id,
                    "sequence": sequence,
                    "text_length": len(text),
                    "audio_size": audio_size
                }
            )
            
            return audio_base64

        except Exception as e:
            duration_ms = round((datetime.now().timestamp() - start_time) * 1000, 2)
            self.logger.api_call(
                "Edge TTS",
                success=False,
                duration_ms=duration_ms,
                context={
                    "chunk_id": chunk_id,
                    "sequence": sequence,
                    "error": str(e)
                }
            )
            self.logger.error(f"TTS error for chunk {chunk_id}", {
                "error": str(e),
                "sequence": sequence
            })
            return None

    def start_chunk(self, chunk_id: int) -> None:
        """Mark a chunk as active for processing."""
        self.active_chunks.add(chunk_id)
        self.logger.debug("Started chunk processing", {"chunk_id": chunk_id})

    def clear_chunk(self, chunk_id: Optional[int] = None) -> None:
        """Clear specific chunk or all chunks."""
        if chunk_id is not None:
            self.active_chunks.discard(chunk_id)
            self.logger.debug("Cleared specific chunk", {"chunk_id": chunk_id})
        else:
            self.active_chunks.clear()
            self.logger.debug("Cleared all chunks")
