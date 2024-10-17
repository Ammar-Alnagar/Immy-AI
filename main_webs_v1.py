import os
import time
import requests
import speech_recognition as sr
from typing import IO
from io import BytesIO
import asyncio
import websockets
import json
import base64
import pygame
from dotenv import load_dotenv
import tkinter as tk
from tkinter import messagebox

# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from environment variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
LLMINABOX_API_URL = os.getenv("LLMINABOX_API_URL")

# WebSocket URL for ElevenLabs real-time TTS
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Adam's pre-made voice
ELEVENLABS_WS_URL = f"wss://api.elevenlabs.io/v1/text-to-speech/{{VOICE_ID}}/stream-input?model_id=eleven_turbo_v2_5"

# Initialize pygame mixer
pygame.mixer.init()

# Function to stream and play audio using pygame
async def stream(audio_stream):
    """Stream audio data using pygame for playback."""
    audio_buffer = BytesIO()  # Create a buffer to hold the incoming audio chunks

    async for chunk in audio_stream:
        if chunk:
            print("Received audio chunk")  # Debugging log
            audio_buffer.write(chunk)  # Write the chunk to the buffer

    # If the buffer is not empty, play the audio
    if audio_buffer.tell() > 0:
        # Reset the buffer to the beginning for playback
        audio_buffer.seek(0)

        # Play the audio using pygame
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
    else:
        print("No audio received to play.")


async def text_to_speech_websocket(text: str) -> IO[bytes]:
    """Send text to ElevenLabs API via WebSockets and stream the returned audio."""
    uri = ELEVENLABS_WS_URL.format(voice_id=VOICE_ID)

    async with websockets.connect(uri) as websocket:
        # Initial request
        bos_message = {
            "text": " ",  # Start of stream (blank or single space)
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8
            },
            "xi_api_key": ELEVENLABS_API_KEY,  # Using API key
        }
        await websocket.send(json.dumps(bos_message))

        # Send actual text input
        input_message = {
            "text": text
        }
        await websocket.send(json.dumps(input_message))

        # End of stream message
        eos_message = {
            "text": ""  # Empty string as per documentation
        }
        await websocket.send(json.dumps(eos_message))

        # Receive and handle the audio data
        async def audio_data_stream():
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)

                    if "audio" in data:
                        # If audio data is present, decode it
                        audio_chunk = base64.b64decode(data["audio"])
                        yield audio_chunk
                    else:
                        break
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break

        # Stream the audio data received from ElevenLabs
        await stream(audio_data_stream())


# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        except Exception as e:
            print(f"Error: {str(e)}")
            return None


# Function to send text to LLMinaBox API and get the response
def send_to_LLMinBox(user_input):
    payload = {"question": user_input}
    try:
        response = requests.post(LLMINABOX_API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes

        print(f"Response status code: {response.status_code}")
        
        # Try to parse the JSON response
        json_response = response.json()
        
        # Extract the text from the response
        response_text = json_response.get('text', 'No text field in JSON')
        return response_text
    except requests.exceptions.RequestException as req_err:
        print(f"Request to LLMinaBox failed: {req_err}")
        return f"Error: Failed to connect to LLMinaBox. {str(req_err)}"


# Function triggered by the Tkinter button to start the process
def start_recording():
    user_input = recognize_speech()
    if user_input:
        response_text = send_to_LLMinBox(user_input)
        print("LLMinaBox response:", response_text)
        if not response_text.startswith("Error:"):
            # Use asyncio to run the text-to-speech websocket stream
            loop = asyncio.get_event_loop()
            try:
                loop.run_until_complete(text_to_speech_websocket(response_text))
            except Exception as e:
                print(f"Error during TTS WebSocket handling: {str(e)}")
        else:
            print("Skipping text-to-speech due to error in LLMinaBox response")
            messagebox.showerror("Error", "LLMinaBox response error")


# Create the Tkinter UI
def create_gui():
    window = tk.Tk()
    window.title("Speech Recognition App")

    # Create and place the button on the window
    record_button = tk.Button(window, text="Start Recording", command=start_recording, padx=20, pady=10)
    record_button.pack(pady=20)

    # Start the Tkinter main loop
    window.mainloop()


if __name__ == "__main__":
    create_gui()
