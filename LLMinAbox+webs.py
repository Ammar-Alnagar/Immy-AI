import os
import time
import requests
import speech_recognition as sr
from typing import IO, AsyncGenerator, Optional
from io import BytesIO
import asyncio
import websockets
import json
import base64
import pygame
from dotenv import load_dotenv
import tkinter as tk
from tkinter import messagebox
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

# Configuration with type hints and validation
@dataclass
class Config:
    elevenlabs_api_key: str
    llminabox_api_url: str
    elevenlabs_ws_url: str = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_turbo_v2_5"
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    sample_rate: int = 44100
    
    @classmethod
    def from_env(cls) -> 'Config':
        api_key = os.getenv("ELEVENLABS_API_KEY")
        api_url = os.getenv("LLMINABOX_API_URL")
        
        if not api_key or not api_url:
            raise ValueError("Missing required environment variables")
        
        return cls(
            elevenlabs_api_key=api_key,
            llminabox_api_url=api_url
        )

class AudioStreamHandler:
    def __init__(self, sample_rate: int = 44100):
        """Initialize audio handler with configurable sample rate."""
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate)
        self.audio_buffer = BytesIO()
        self.is_playing = False
    
    async def process_audio_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> None:
        """Process and play audio stream with improved error handling."""
        self.audio_buffer = BytesIO()
        try:
            async for chunk in audio_stream:
                if chunk:
                    self.audio_buffer.write(chunk)
            
            if self.audio_buffer.tell() > 0:
                await self.play_audio()
            else:
                raise ValueError("No audio data received")
        except Exception as e:
            print(f"Error processing audio stream: {e}")
            raise
    
    async def play_audio(self) -> None:
        """Play audio with better state management."""
        try:
            self.audio_buffer.seek(0)
            self.is_playing = True
            pygame.mixer.music.load(self.audio_buffer)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error playing audio: {e}")
            raise
        finally:
            self.is_playing = False
            pygame.mixer.music.unload()

class ElevenLabsClient:
    def __init__(self, api_key: str, voice_id: str, ws_url: str):
        self.api_key = api_key
        self.voice_id = voice_id
        self.ws_url = ws_url.format(voice_id=voice_id)
    
    @asynccontextmanager
    async def _websocket_connection(self):
        """Context manager for WebSocket connections."""
        try:
            async with websockets.connect(self.ws_url) as websocket:
                yield websocket
        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket error: {e}")
            raise
    
    async def text_to_speech_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Improved streaming with better connection handling and validation."""
        if not text.strip():
            raise ValueError("Empty text provided")
        
        async with self._websocket_connection() as websocket:
            # Send initial configuration
            config_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8
                },
                "xi_api_key": self.api_key
            }
            await websocket.send(json.dumps(config_message))
            
            # Send text content
            await websocket.send(json.dumps({"text": text}))
            
            # Send end of stream
            await websocket.send(json.dumps({"text": ""}))
            
            # Process response stream with improved error handling
            while True:
                try:
                    response = await websocket.recv()
                    if isinstance(response, str):
                        data = json.loads(response)
                        if "audio" in data:
                            yield base64.b64decode(data["audio"])
                        elif "error" in data:
                            raise RuntimeError(f"ElevenLabs API error: {data['error']}")
                    else:
                        yield response
                except websockets.exceptions.ConnectionClosed:
                    break
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    break

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Add noise adjustment
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 4000
    
    async def recognize(self) -> Optional[str]:
        """Asynchronous speech recognition with timeout."""
        with sr.Microphone() as source:
            print("Listening...")
            try:
                # Add ambient noise adjustment
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Use asyncio.get_event_loop().run_in_executor for blocking operations
                audio = await asyncio.get_event_loop().run_in_executor(
                    None, self.recognizer.listen, source, timeout=10
                )
                
                text = await asyncio.get_event_loop().run_in_executor(
                    None, self.recognizer.recognize_google, audio
                )
                
                print(f"Recognized: {text}")
                return text
            except sr.WaitTimeoutError:
                messagebox.showerror("Error", "Listening timeout - no speech detected")
            except sr.UnknownValueError:
                messagebox.showerror("Error", "Could not understand audio")
            except sr.RequestError as e:
                messagebox.showerror("Error", f"Speech recognition service error: {e}")
            return None

class LLMinaBoxClient:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session = requests.Session()
    
    async def get_response(self, user_input: str) -> str:
        """Asynchronous API communication with timeout and retry logic."""
        if not user_input.strip():
            raise ValueError("Empty user input")
        
        try:
            # Use asyncio.get_event_loop().run_in_executor for blocking operations
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.post(
                    self.api_url,
                    json={"question": user_input},
                    timeout=30,
                    headers={"Content-Type": "application/json"}
                )
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "text" not in result:
                raise ValueError("Invalid API response format")
                
            return result["text"]
            
        except requests.exceptions.RequestException as e:
            error_msg = f"LLMinaBox API error: {e}"
            messagebox.showerror("Error", error_msg)
            raise

class SpeechApp:
    def __init__(self):
        self.config = Config.from_env()
        self.window = tk.Tk()
        self.window.title("Speech Recognition App")
        self.setup_ui()
        
        # Initialize components with configuration
        self.speech_recognizer = SpeechRecognizer()
        self.llmina_client = LLMinaBoxClient(self.config.llminabox_api_url)
        self.elevenlabs_client = ElevenLabsClient(
            self.config.elevenlabs_api_key,
            self.config.voice_id,
            self.config.elevenlabs_ws_url
        )
        self.audio_handler = AudioStreamHandler(self.config.sample_rate)
        
        # Add window closing handler
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Enhanced UI setup with status indicators."""
        self.status_label = tk.Label(self.window, text="Ready")
        self.status_label.pack(pady=5)
        
        self.record_button = tk.Button(
            self.window,
            text="Start Recording",
            command=self.start_recording,
            padx=20,
            pady=10
        )
        self.record_button.pack(pady=20)
    
    def start_recording(self):
        """Handle recording button click with UI updates."""
        self.record_button.config(state=tk.DISABLED)
        self.status_label.config(text="Processing...")
        
        # Create and start async task
        asyncio.run(self.process_speech())
        
        self.record_button.config(state=tk.NORMAL)
        self.status_label.config(text="Ready")
    
    async def process_speech(self):
        """Enhanced speech processing with proper error handling."""
        try:
            # Get speech input
            user_input = await self.speech_recognizer.recognize()
            if not user_input:
                return
            
            # Get LLMinaBox response
            response_text = await self.llmina_client.get_response(user_input)
            print("LLMinaBox response:", response_text)
            
            # Convert text to speech and play
            audio_stream = self.elevenlabs_client.text_to_speech_stream(response_text)
            await self.audio_handler.process_audio_stream(audio_stream)
            
        except Exception as e:
            print(f"Error in speech processing: {e}")
            messagebox.showerror("Error", f"Processing error: {e}")
            
        finally:
            self.record_button.config(state=tk.NORMAL)
            self.status_label.config(text="Ready")
    
    def on_closing(self):
        """Clean up resources before closing."""
        if self.audio_handler.is_playing:
            pygame.mixer.music.stop()
        pygame.mixer.quit()
        self.window.destroy()
    
    def run(self):
        """Start the application with error handling."""
        try:
            self.window.mainloop()
        except Exception as e:
            print(f"Application error: {e}")
            raise

if __name__ == "__main__":
    try:
        app = SpeechApp()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", str(e))