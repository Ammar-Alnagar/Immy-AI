import os
import time
import requests
import speech_recognition as sr
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import pygame
import RPi.GPIO as GPIO

# Load environment variables from .env file


# Retrieve the API keys from environment variables
ELEVENLABS_API_KEY = "sk_b917a35288dd727edb8ebc70744f4fc9c4cf86daa3bf1036"
LLMINABOX_API_URL = "https://llminabox.criticalfutureglobal.com/api/v1/prediction/40e5d309-37b7-47c3-8da4-66e606bc350c"

# Initialize Eleven Labs client
eleven_labs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# GPIO setup for button
BUTTON_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Function to convert text to speech and return as audio stream
def text_to_speech_stream(text: str) -> IO[bytes]:
    start_time = time.time()
    # Perform the text-to-speech conversion
    response = eleven_labs_client.text_to_speech.convert_as_stream(
        voice_id="jBpfuIE2acCO8z3wKNLl",  # Adam pre-made voice
        output_format="mp3_22050_32",
        optimize_streaming_latency="4",
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

# Function to say "Hi there" when button is pressed
def say_hi():
    hi_stream = text_to_speech_stream("Hi there!")
    play_audio(hi_stream)

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("Listening...")
        try:
            # Listen with timeout
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        
        except sr.WaitTimeoutError:
            print("Listening timed out")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

# Function to send text to LLMinaBox API and get the response
def send_to_LLMinBox(user_input):
    payload = {"question": user_input}
    try:
        response = requests.post(LLMINABOX_API_URL, json=payload, stream=True)
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

# Main loop to wait for button press and process the input
def main():
    print("Waiting for button press...")

    while True:
        # Detect button press (falling edge)
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            # Say "Hi there" first
            say_hi()
            
            # Small delay to ensure "Hi there" is fully spoken
            time.sleep(1)
            
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

        time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted by user")
    finally:
        GPIO.cleanup()