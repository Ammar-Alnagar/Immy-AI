import os
import time
import asyncio
import websockets
import json
import speech_recognition as sr
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import pygame
import tkinter as tk
from tkinter import messagebox
import threading

# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from environment variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
LLMINABOX_WS_URL = os.getenv("LLMINABOX_WS_URL")  # WebSocket URL for LLMinaBox

# Initialize Eleven Labs client
eleven_labs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    async def text_to_speech_stream(self, text: str) -> IO[bytes]:
        # Convert to speech using async pattern
        response = await asyncio.to_thread(
            eleven_labs_client.text_to_speech.convert,
            voice_id="jBpfuIE2acCO8z3wKNLl",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            )
        )

        audio_stream = BytesIO()
        for chunk in response:
            if chunk:
                audio_stream.write(chunk)
        audio_stream.seek(0)
        return audio_stream

    async def play_audio(self, audio_stream):
        # Initialize pygame mixer in the main thread
        await asyncio.to_thread(pygame.mixer.init)
        
        # Load and play audio
        await asyncio.to_thread(pygame.mixer.music.load, audio_stream)
        await asyncio.to_thread(pygame.mixer.music.play)
        
        # Wait for audio to finish
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)

class WebSocketClient:
    def __init__(self, url, gui):
        self.url = url
        self.gui = gui
        self.websocket = None
        self.audio_processor = AudioProcessor()
        self.is_connected = False

    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.url)
            self.is_connected = True
            self.gui.update_connection_status("Connected to LLMinaBox")
            return True
        except Exception as e:
            self.gui.update_connection_status(f"Connection failed: {str(e)}")
            return False

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            self.gui.update_connection_status("Disconnected")

    async def send_message(self, message):
        if not self.is_connected:
            self.gui.update_status("Not connected to LLMinaBox")
            return

        try:
            # Send message to LLMinaBox
            await self.websocket.send(json.dumps({"question": message}))
            
            # Wait for response
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            # Process response
            response_text = response_data.get('text', 'No response received')
            self.gui.update_status(f"Response: {response_text}")
            
            # Convert to speech and play
            audio_stream = await self.audio_processor.text_to_speech_stream(response_text)
            await self.audio_processor.play_audio(audio_stream)
            
        except Exception as e:
            self.gui.update_status(f"Error: {str(e)}")

class GUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("LLMinaBox Speech Recognition")
        self.setup_ui()
        self.client = WebSocketClient(LLMINABOX_WS_URL, self)
        self.recognizer = sr.Recognizer()
        self.is_recording = False

    def setup_ui(self):
        # Status labels
        self.connection_label = tk.Label(self.window, text="Disconnected")
        self.connection_label.pack(pady=5)
        
        self.status_label = tk.Label(self.window, text="Ready")
        self.status_label.pack(pady=5)

        # Buttons
        self.connect_button = tk.Button(
            self.window,
            text="Connect",
            command=self.toggle_connection,
            padx=20,
            pady=10
        )
        self.connect_button.pack(pady=5)

        self.record_button = tk.Button(
            self.window,
            text="Start Recording",
            command=self.toggle_recording,
            state=tk.DISABLED,
            padx=20,
            pady=10
        )
        self.record_button.pack(pady=5)

    def update_connection_status(self, status):
        self.connection_label.config(text=status)
        if "Connected" in status:
            self.record_button.config(state=tk.NORMAL)
        else:
            self.record_button.config(state=tk.DISABLED)

    def update_status(self, status):
        self.status_label.config(text=status)

    def toggle_connection(self):
        if not self.client.is_connected:
            asyncio.run(self.client.connect())
            if self.client.is_connected:
                self.connect_button.config(text="Disconnect")
        else:
            asyncio.run(self.client.disconnect())
            self.connect_button.config(text="Connect")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.config(text="Stop Recording")
        self.update_status("Listening...")
        
        # Start recording in a separate thread
        threading.Thread(target=self.record_audio).start()

    def stop_recording(self):
        self.is_recording = False
        self.record_button.config(text="Start Recording")
        self.update_status("Ready")

    def record_audio(self):
        try:
            with sr.Microphone() as source:
                audio = self.recognizer.listen(source)
                text = self.recognizer.recognize_google(audio)
                self.update_status(f"Recognized: {text}")
                
                # Send to LLMinaBox
                asyncio.run(self.client.send_message(text))
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
        finally:
            self.stop_recording()

    def run(self):
        self.window.mainloop()

def main():
    gui = GUI()
    gui.run()

if __name__ == "__main__":
    # Initialize pygame
    pygame.init()
    main()