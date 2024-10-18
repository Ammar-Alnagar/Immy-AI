import os
import time
import requests
import speech_recognition as sr
from typing import IO
from io import BytesIO
import tkinter as tk
from tkinter import messagebox
import json
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3

class SpeechBot:
    def __init__(self):
        self.OLLAMA_API_URL = "http://localhost:11434/api/chat"  # Changed to chat endpoint
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

    def text_to_speech(self, text: str) -> None:
        start_time = time.time()
        temp_file = "temp_speech.wav"
        self.engine.save_to_file(text, temp_file)
        self.engine.runAndWait()
        
        try:
            audio = AudioSegment.from_wav(temp_file)
            play(audio)
        except Exception as e:
            print(f"Error playing audio: {str(e)}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        end_time = time.time()
        print(f"Text-to-speech conversion and playback took {end_time - start_time:.2f} seconds")

    def recognize_speech(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            start_time = time.time()
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                end_time = time.time()
                print(f"Recognized: {text}")
                print(f"Speech recognition took {end_time - start_time:.2f} seconds")
                return text
            except Exception as e:
                print(f"Error: {str(e)}")
                return None

    def send_to_ollama(self, user_input):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Immy, a magical AI-powered teddy bear who loves to chat with children. "
                    "You are kind, funny, and full of wonder, always ready to tell stories, answer questions, "
                    "and offer friendly advice. When speaking, you are playful, patient, and use simple, "
                    "child-friendly language. You encourage curiosity, learning, and imagination. "
                    "Keep your responses relatively brief and engaging."
                )
            },
            {
                "role": "user",
                "content": user_input
            }
        ]

        start_time = time.time()
        
        payload = {
            "model": "qwen2.5:0.5b-instruct-q8_0",
            "messages": messages,
            "stream": False
        }

        try:
            response = requests.post(self.OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            response_text = response_data.get('message', {}).get('content', '')
            
            end_time = time.time()
            print(f"Ollama API response took {end_time - start_time:.2f} seconds")
            
            return response_text
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama API: {str(e)}")
            return None

    def start_recording(self):
        user_input = self.recognize_speech()
        if user_input:
            response_text = self.send_to_ollama(user_input)
            print("Ollama response:", response_text)
            if response_text:
                self.text_to_speech(response_text)
            else:
                print("No response from Ollama.")
                messagebox.showerror("Error", "No response from Ollama API")

    def create_gui(self):
        self.window = tk.Tk()
        self.window.title("Immy - Your AI Teddy Bear Friend")
        self.window.geometry("400x300")
        
        frame = tk.Frame(self.window, padx=20, pady=20)
        frame.pack(expand=True)
        
        title_label = tk.Label(
            frame, 
            text="Talk with Immy",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=20)
        
        instructions = tk.Label(
            frame,
            text="Press the button and start speaking!\nImmy is excited to chat with you!",
            font=("Arial", 12),
            justify=tk.CENTER
        )
        instructions.pack(pady=20)
        
        record_button = tk.Button(
            frame,
            text="Start Talking",
            command=self.start_recording,
            font=("Arial", 14),
            padx=20,
            pady=10,
            bg="#4CAF50",
            fg="white"
        )
        record_button.pack(pady=20)
        
        self.window.mainloop()

if __name__ == "__main__":
    bot = SpeechBot()
    bot.create_gui()