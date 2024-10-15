import os
import time
import requests
import speech_recognition as sr
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from groq import Groq
from dotenv import load_dotenv
import pygame

# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

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

    end_time = time.time()
    print(f"Text-to-speech conversion took {end_time - start_time:.2f} seconds")

    # Return the stream for further use
    return audio_stream

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
    # Define the system prompt to guide the model's behavior
    system_prompt = (
        "You are Immy, a magical AI-powered teddy bear who loves to chat with children. "
        "You are kind, funny, and full of wonder, always ready to tell stories, answer questions, and offer friendly advice. "
        "When speaking, you are playful, patient, and use simple, child-friendly language. You encourage curiosity, learning, and imagination."
    )

    start_time = time.time()
    # Send the prompt and user message to Groq API
    chat_response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # Use your preferred model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
    )
    end_time = time.time()
    print(f"Groq API response took {end_time - start_time:.2f} seconds")

    # Return the response from Groq
    return chat_response.choices[0].message.content

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
            response_text = send_to_groq(user_input)
            print("Groq response:", response_text)
            audio_stream = text_to_speech_stream(response_text)
            play_audio(audio_stream)

if __name__ == "__main__":
    main()