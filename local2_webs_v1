import os
import time
import json
import asyncio
import websockets
import requests
import speech_recognition as sr
from typing import IO
from io import BytesIO
from groq import Groq
from dotenv import load_dotenv
import pygame
import tkinter as tk
from tkinter import messagebox

# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Websocket-based text to speech function
async def text_to_speech_ws_streaming(text: str) -> BytesIO:
    voice_id = "jBpfuIE2acCO8z3wKNLl"  # Adam pre-made voice
    model_id = "eleven_turbo_v2"
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"

    audio_stream = BytesIO()
    start_time = time.time()

    try:
        async with websockets.connect(uri) as websocket:
            # Send initial configuration
            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {
                    "stability": 0.0,
                    "similarity_boost": 1.0,
                    "style": 0.0,
                    "use_speaker_boost": True
                },
                "generation_config": {
                    "chunk_length_schedule": [120, 160, 250, 290]
                },
                "xi_api_key": ELEVENLABS_API_KEY,
            }))

            # Send the actual text
            await websocket.send(json.dumps({"text": text}))

            # Send empty string to close the connection
            await websocket.send(json.dumps({"text": ""}))

            # Receive and process audio chunks
            while True:
                try:
                    chunk = await websocket.recv()
                    if isinstance(chunk, bytes):
                        audio_stream.write(chunk)
                except websockets.exceptions.ConnectionClosed:
                    break

        end_time = time.time()
        print(f"Text-to-speech conversion took {end_time - start_time:.2f} seconds")

        # Reset stream position
        audio_stream.seek(0)
        return audio_stream

    except Exception as e:
        print(f"Error in websocket streaming: {str(e)}")
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

# Function to send text to Groq API and get the response
def send_to_groq(user_input):
    system_prompt = (
        "You are Immy, a magical AI-powered teddy bear who loves to chat with children. "
        "You are kind, funny, and full of wonder, always ready to tell stories, answer questions, and offer friendly advice. "
        "When speaking, you are playful, patient, and use simple, child-friendly language. You encourage curiosity, learning, and imagination."
    )

    start_time = time.time()
    chat_response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
    )
    end_time = time.time()
    print(f"Groq API response took {end_time - start_time:.2f} seconds")

    return chat_response.choices[0].message.content

# Function to play audio from a BytesIO stream
def play_audio(audio_stream):
    if audio_stream is None:
        print("No audio stream to play")
        return

    pygame.mixer.init()
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# Function to handle the async text-to-speech conversion
async def process_speech_async(response_text):
    audio_stream = await text_to_speech_ws_streaming(response_text)
    if audio_stream:
        play_audio(audio_stream)

# Modified button callback to handle async operations
def start_recording():
    user_input = recognize_speech()
    if user_input:
        response_text = send_to_groq(user_input)
        print("Groq response:", response_text)
        if response_text:
            # Create and run the async event loop
            asyncio.run(process_speech_async(response_text))
        else:
            print("No response from Groq.")
            messagebox.showerror("Error", "No response from Groq API")

# Create the Tkinter UI
def create_gui():
    window = tk.Tk()
    window.title("Speech Recognition with Groq & Eleven Labs")

    record_button = tk.Button(window, text="Start Recording", command=start_recording, padx=20, pady=10)
    record_button.pack(pady=20)

    window.mainloop()

if __name__ == "__main__":
    create_gui()