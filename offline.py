import os
import time
import requests
import tkinter as tk
from tkinter import messagebox
import json
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import threading
from queue import Queue
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class SpeechBot:
    def __init__(self):
        self.OLLAMA_API_URL = "http://localhost:11434/api/chat"
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        # Initialize Faster Whisper with optimized settings
        print("Loading Whisper model...")
        self.model = WhisperModel(
            "turbo",
            device="cpu" if self.is_cuda_available() else "cpu",
            compute_type="int8" if self.is_cuda_available() else "int8",
            cpu_threads=8,  # Adjust based on your CPU
            num_workers=4   # Parallel processing workers
        )
        print("Whisper model loaded!")
        
        # Audio recording parameters
        self.CHUNK = 2048  # Larger chunk size for better performance
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Setup voice
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        
        # Create queues for async processing
        self.audio_queue = Queue()
        self.response_queue = Queue()

    def is_cuda_available(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def text_to_speech(self, text: str) -> None:
        """Optimized text-to-speech conversion"""
        if not text:
            return
            
        temp_file = "temp_speech.wav"
        try:
            self.engine.save_to_file(text, temp_file)
            self.engine.runAndWait()
            audio = AudioSegment.from_wav(temp_file)
            play(audio)
        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def record_audio(self, duration=3):
        """Optimized audio recording"""
        frames = []
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("Recording...")
        
        # Calculate total chunks needed
        total_chunks = int((self.RATE * duration) / self.CHUNK)
        
        # Record audio
        for _ in range(total_chunks):
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))
            except Exception as e:
                print(f"Error recording: {str(e)}")
        
        print("Recording finished.")
        
        stream.stop_stream()
        stream.close()
        
        return np.concatenate(frames, axis=0)

    def recognize_speech(self):
        """Optimized speech recognition"""
        try:
            # Record audio
            audio_data = self.record_audio(duration=3)  # Reduced duration for faster response
            
            # Run Whisper inference
            segments, _ = self.model.transcribe(
                audio_data,
                language='en',
                beam_size=1,  # Reduced beam size for speed
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            
            # Get the transcribed text
            recognized_text = " ".join([segment.text for segment in segments]).strip()
            
            if recognized_text:
                print(f"Recognized: {recognized_text}")
                return recognized_text
            return None
            
        except Exception as e:
            print(f"Error in speech recognition: {str(e)}")
            return None

    def process_ollama_response(self, user_input):
        """Process Ollama response in a separate thread"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Immy, a magical AI-powered teddy bear who loves to chat with children. "
                    "You are kind, funny, and full of wonder, always ready to tell stories, answer questions, "
                    "and offer friendly advice. When speaking, you are playful, patient, and use simple, "
                    "child-friendly language. You encourage curiosity, learning, and imagination."
                    "Dont use emojis in your responses. "
                )
            },
            {
                "role": "user",
                "content": user_input
            }
        ]

        payload = {
            "model": "qwen2.5:0.5b",
            "messages": messages,
            "stream": False
        }

        try:
            response = requests.post(self.OLLAMA_API_URL, json=payload, timeout=5)
            response.raise_for_status()
            response_text = response.json().get('message', {}).get('content', '')
            self.response_queue.put(response_text)
        except Exception as e:
            print(f"Error with Ollama API: {str(e)}")
            self.response_queue.put(None)

    def start_recording(self):
        """Handle recording and response generation"""
        # Update button state
        self.record_button.config(state='disabled', text="Listening...")
        self.window.update()
        
        # Get speech input
        user_input = self.recognize_speech()
        
        if user_input:
            # Start Ollama processing in separate thread
            threading.Thread(
                target=self.process_ollama_response,
                args=(user_input,),
                daemon=True
            ).start()
            
            # Wait for response with timeout
            try:
                response_text = self.response_queue.get(timeout=10)
                if response_text:
                    self.text_to_speech(response_text)
                else:
                    messagebox.showerror("Error", "No response received")
            except Exception:
                messagebox.showerror("Error", "Response timeout")
        
        # Reset button state
        self.record_button.config(state='normal', text="Start Talking")
        self.window.update()

    def create_gui(self):
        """Create the GUI with improved responsiveness"""
        self.window = tk.Tk()
        self.window.title("Immy - Your AI Teddy Bear Friend")
        self.window.geometry("400x300")
        
        frame = tk.Frame(self.window, padx=20, pady=20)
        frame.pack(expand=True)
        
        tk.Label(
            frame, 
            text="Talk with Immy",
            font=("Arial", 18, "bold")
        ).pack(pady=20)
        
        tk.Label(
            frame,
            text="Press the button and start speaking!\nImmy is excited to chat with you!",
            font=("Arial", 12),
            justify=tk.CENTER
        ).pack(pady=20)
        
        self.record_button = tk.Button(
            frame,
            text="Start Talking",
            command=lambda: threading.Thread(target=self.start_recording, daemon=True).start(),
            font=("Arial", 14),
            padx=20,
            pady=10,
            bg="#4CAF50",
            fg="white"
        )
        self.record_button.pack(pady=20)
        
        self.window.mainloop()

    def cleanup(self):
        """Cleanup resources"""
        self.audio.terminate()
        self.engine.stop()

if __name__ == "__main__":
    bot = SpeechBot()
    try:
        bot.create_gui()
    finally:
        bot.cleanup()