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
LLMINABOX_WS_URL = os.getenv("LLMINABOX_WS_URL")

# Initialize Eleven Labs client
eleven_labs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    async def text_to_speech_stream(self, text: str) -> IO[bytes]:
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
        await asyncio.to_thread(pygame.mixer.init)
        await asyncio.to_thread(pygame.mixer.music.load, audio_stream)
        await asyncio.to_thread(pygame.mixer.music.play)
        
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)

class WebSocketClient:
    def __init__(self, url, gui):
        self.url = url
        self.gui = gui
        self.websocket = None
        self.audio_processor = AudioProcessor()
        self.is_connected = False
        self.reconnect_delay = 1  # Start with 1 second delay
        self.max_reconnect_delay = 30  # Maximum delay between reconnection attempts
        self.should_reconnect = True

    async def connect_with_retry(self):
        while self.should_reconnect:
            try:
                if not self.is_connected:
                    self.gui.update_status("Connecting to LLMinaBox...")
                    self.websocket = await websockets.connect(self.url)
                    self.is_connected = True
                    self.reconnect_delay = 1  # Reset delay on successful connection
                    self.gui.update_status("Connected to LLMinaBox")
                    self.gui.enable_recording()
                    
                    # Start heartbeat
                    asyncio.create_task(self.heartbeat())
                    
                    # Wait for connection to close
                    await self.websocket.wait_closed()
                    
                    # Connection closed, update status
                    self.is_connected = False
                    self.gui.update_status("Connection lost. Reconnecting...")
                    self.gui.disable_recording()
                    
            except Exception as e:
                self.is_connected = False
                self.gui.update_status(f"Connection failed. Retrying in {self.reconnect_delay}s...")
                self.gui.disable_recording()
                
                await asyncio.sleep(self.reconnect_delay)
                
                # Exponential backoff with maximum delay
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    async def heartbeat(self):
        while self.is_connected:
            try:
                await self.websocket.ping()
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except:
                break

    async def send_message(self, message):
        if not self.is_connected:
            self.gui.update_status("Not connected to LLMinaBox")
            return

        try:
            await self.websocket.send(json.dumps({"question": message}))
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            response_text = response_data.get('text', 'No response received')
            self.gui.update_status(f"Response: {response_text}")
            
            audio_stream = await self.audio_processor.text_to_speech_stream(response_text)
            await self.audio_processor.play_audio(audio_stream)
            
        except Exception as e:
            self.gui.update_status(f"Error: {str(e)}")

    async def cleanup(self):
        self.should_reconnect = False
        if self.websocket:
            await self.websocket.close()

class GUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("LLMinaBox Speech Recognition")
        self.setup_ui()
        self.client = WebSocketClient(LLMINABOX_WS_URL, self)
        self.recognizer = sr.Recognizer()
        self.is_recording = False
        
        # Start WebSocket connection in a separate thread
        self.start_websocket_thread()

    def setup_ui(self):
        self.status_label = tk.Label(self.window, text="Initializing...", wraplength=300)
        self.status_label.pack(pady=10)

        self.record_button = tk.Button(
            self.window,
            text="Start Recording",
            command=self.toggle_recording,
            state=tk.DISABLED,
            padx=20,
            pady=10
        )
        self.record_button.pack(pady=10)

    def start_websocket_thread(self):
        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.client.connect_with_retry())

        threading.Thread(target=run_async_loop, daemon=True).start()

    def update_status(self, status):
        self.status_label.config(text=status)

    def enable_recording(self):
        self.record_button.config(state=tk.NORMAL)

    def disable_recording(self):
        self.record_button.config(state=tk.DISABLED)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.config(text="Stop Recording")
        self.update_status("Listening...")
        threading.Thread(target=self.record_audio, daemon=True).start()

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
        # Handle window closing
        def on_closing():
            asyncio.run(self.client.cleanup())
            self.window.destroy()
            
        self.window.protocol("WM_DELETE_WINDOW", on_closing)
        self.window.mainloop()

def main():
    pygame.init()
    gui = GUI()
    gui.run()

if __name__ == "__main__":
    main()