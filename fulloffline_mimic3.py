import os
import time
import requests
import speech_recognition as sr
from typing import IO
from io import BytesIO
import pygame
import tkinter as tk
from tkinter import messagebox
import json
from urllib.parse import quote

# Function to convert text to speech using Mimic 3
def text_to_speech_stream(text: str) -> IO[bytes]:
    start_time = time.time()
    
    # Mimic 3 server URL - adjust if running on different port
    MIMIC3_URL = "http://localhost:59125/api/tts"
    
    # Parameters for Mimic 3
    params = {
        "text": text,
        "voice": "en_US/vctk_low#p226",  # You can change the voice
        "noiseScale": 0.667,
        "noiseW": 0.8,
        "lengthScale": 1.0,
        "ssml": "false"
    }
    
    try:
        # Make request to Mimic 3 server
        response = requests.get(
            f"{MIMIC3_URL}?text={quote(text)}&voice={params['voice']}&noiseScale={params['noiseScale']}&noiseW={params['noiseW']}&lengthScale={params['lengthScale']}&ssml={params['ssml']}",
            stream=True
        )
        response.raise_for_status()
        
        # Create a BytesIO object to hold the audio data
        audio_stream = BytesIO()
        
        # Write the audio data to the stream
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                audio_stream.write(chunk)
        
        # Reset stream position
        audio_stream.seek(0)
        
        end_time = time.time()
        print(f"Text-to-speech conversion took {end_time - start_time:.2f} seconds")
        
        return audio_stream
    
    except requests.exceptions.RequestException as e:
        print(f"Error in text-to-speech conversion: {str(e)}")
        return None

# Function to recognize speech
def recognize_speech():
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

# Function to send text to Ollama API and get the response
def send_to_ollama(user_input):
    # Define the system prompt to guide the model's behavior
    system_prompt = (
        "You are Immy, a magical AI-powered teddy bear who loves to chat with children. "
        "You are kind, funny, and full of wonder, always ready to tell stories, answer questions, and offer friendly advice. "
        "When speaking, you are playful, patient, and use simple, child-friendly language. You encourage curiosity, learning, and imagination."
    )

    start_time = time.time()
    
    # Prepare the request payload for Ollama
    payload = {
        "model": "llama2",  # You can change this to any model you have pulled in Ollama
        "prompt": f"System: {system_prompt}\n\nUser: {user_input}\n\nAssistant:",
        "stream": False
    }

    try:
        # Send request to Ollama API
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        
        # Parse the response
        response_data = response.json()
        response_text = response_data.get('response', '')
        
        end_time = time.time()
        print(f"Ollama API response took {end_time - start_time:.2f} seconds")
        
        return response_text
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama API: {str(e)}")
        return None

# Function to play audio from a BytesIO stream
def play_audio(audio_stream):
    if audio_stream is None:
        print("No audio stream to play")
        return
        
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Load the audio stream into pygame
    pygame.mixer.music.load(audio_stream)
    
    # Play the audio
    pygame.mixer.music.play()
    
    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# Create a more sophisticated GUI
class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Teddy Bear Chat")
        self.root.geometry("400x500")
        
        # Create main frame
        main_frame = tk.Frame(root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Chat history display
        self.chat_history = tk.Text(main_frame, height=15, width=40, wrap=tk.WORD, state='disabled')
        self.chat_history.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Recording button with status indicator
        self.status_var = tk.StringVar(value="Click to Start Recording")
        self.status_label = tk.Label(main_frame, textvariable=self.status_var)
        self.status_label.pack(pady=(0, 10))
        
        self.record_button = tk.Button(
            main_frame,
            text="🎤 Record",
            command=self.start_recording,
            width=20,
            height=2,
            bg="#4CAF50",
            fg="white"
        )
        self.record_button.pack()

    def update_chat_history(self, speaker, text):
        self.chat_history.configure(state='normal')
        self.chat_history.insert(tk.END, f"{speaker}: {text}\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.configure(state='disabled')

    def start_recording(self):
        self.status_var.set("Listening...")
        self.record_button.configure(state='disabled')
        self.root.update()
        
        user_input = recognize_speech()
        if user_input:
            self.update_chat_history("You", user_input)
            self.status_var.set("Processing response...")
            self.root.update()
            
            response_text = send_to_ollama(user_input)
            if response_text:
                self.update_chat_history("Teddy", response_text)
                self.status_var.set("Speaking...")
                self.root.update()
                
                audio_stream = text_to_speech_stream(response_text)
                play_audio(audio_stream)
            else:
                messagebox.showerror("Error", "No response from Ollama API")
        
        self.status_var.set("Click to Start Recording")
        self.record_button.configure(state='normal')

def main():
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()