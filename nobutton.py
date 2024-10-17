import os
import time
import requests
import speech_recognition as sr
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import pygame


# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from environment variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
LLMINABOX_API_URL = os.getenv("LLMINABOX_API_URL")

# Initialize Eleven Labs client
eleven_labs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Function to convert text to speech and return as audio stream
def text_to_speech_stream(text: str) -> IO[bytes]:
    start_time = time.time()
    # Perform the text-to-speech conversion
    response = eleven_labs_client.text_to_speech.convert(
        voice_id="jBpfuIE2acCO8z3wKNLl",  # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Create a BytesIO object to hold the audio data in memory
    audio_stream = BytesIO()

    # Write each chunk of audio data to the stream
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

    # Reset stream position to the beginning
    audio_stream.seek(0)

    # Return the stream for further use
    return audio_stream

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

# Function to play audio from a BytesIO stream
def play_audio(audio_stream):
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Load the audio stream into pygame
    pygame.mixer.music.load(audio_stream)
    
    # Play the audio
    pygame.mixer.music.play()
    
    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# Main loop to keep the application running
def main():
    while True:
        user_input = recognize_speech()
        if user_input:
            response_text = send_to_LLMinBox(user_input)
            print("LLMinaBox response:", response_text)
            if not response_text.startswith("Error:"):
                # Send the response_text directly to ElevenLabs for TTS
                audio_stream = text_to_speech_stream(response_text)
                play_audio(audio_stream)
            else:
                print("Skipping text-to-speech due to error in LLMinaBox response")

if __name__ == "__main__":
    main()