import os
import base64
import tempfile
from openai import AsyncOpenAI
from typing import Optional
from datetime import datetime
from .logger import Logger

class TranscriptionService:
    def __init__(self):
        self.logger = Logger("TranscriptionService")
        self.api_key = self._validate_api_key()
        self.client = self._initialize_openai_client()
        self.logger.info("TranscriptionService initialized successfully")

    def _validate_api_key(self) -> str:
        """Validate and return the OpenAI API key."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.logger.error("OPENAI_API_KEY environment variable is not set")
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.logger.debug("OpenAI API key validated")
            return api_key
        except Exception as e:
            self.logger.error("API key validation failed", {"error": str(e)})
            raise

    def _initialize_openai_client(self) -> AsyncOpenAI:
        """Initialize and test OpenAI client."""
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            self.logger.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            self.logger.error("Failed to initialize OpenAI client", {"error": str(e)})
            raise ValueError(f"Invalid OpenAI API key: {str(e)}")

    async def process_audio_data(self, audio_data: str) -> str:
        """Process base64 encoded audio data and save to temporary file."""
        start_time = datetime.now().timestamp()
        try:
            decoded_audio = base64.b64decode(audio_data)
            self.logger.debug("Audio data decoded", {"size_bytes": len(decoded_audio)})
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(decoded_audio)
                self.logger.debug("Audio saved to temp file", {"path": temp_audio.name})
                self.logger.performance("Audio Processing", start_time, {"size_bytes": len(decoded_audio)})
                return temp_audio.name
        except Exception as e:
            self.logger.error("Failed to process audio data", {"error": str(e)})
            raise RuntimeError(f"Error processing audio data: {str(e)}")

    async def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio file using OpenAI Whisper API."""
        start_time = datetime.now().timestamp()
        try:
            with open(audio_file_path, 'rb') as audio_file:
                self.logger.debug("Starting transcription", {"file": audio_file_path})
                transcript = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                duration_ms = round((datetime.now().timestamp() - start_time) * 1000, 2)
                self.logger.api_call(
                    "Whisper Transcription",
                    success=True,
                    duration_ms=duration_ms,
                    context={"text_length": len(transcript.text)}
                )
                return transcript.text
        except Exception as e:
            duration_ms = round((datetime.now().timestamp() - start_time) * 1000, 2)
            self.logger.api_call(
                "Whisper Transcription",
                success=False,
                duration_ms=duration_ms,
                context={"error": str(e)}
            )
            raise RuntimeError(f"Transcription error: {str(e)}")
        finally:
            try:
                os.unlink(audio_file_path)
                self.logger.debug("Cleaned up temporary files", {"file": audio_file_path})
            except Exception as e:
                self.logger.error("Failed to clean up temporary files", {"error": str(e), "file": audio_file_path})
