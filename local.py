import os
import requests
import speech_recognition as sr
import time
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
from mistralai import Mistral
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Check if the API key is loaded
if "MISTRAL_API_KEY" not in os.environ:
    print("MISTRAL_API_KEY is not set. Please check your .env file.")
    exit(1)

# Constants
MISTRAL_MODEL = "mistral-small-latest"  # Use the specified Mistral model
ELEVEN_LABS_API_KEY = 'sk_482ee3f5c997da5dc21b63628d96b27e81a3a17dcfc5e8bf'
ELEVEN_LABS_VOICE_ID = 'jBpfuIE2acCO8z3wKNLl'

# Initialize Mistral client
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

def recognize_speech():
    """Capture audio from the microphone and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)  # Add a timeout to prevent hanging
            print("Audio captured.")
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

def send_to_Mistral(user_input):
    """Send recognized speech to Mistral AI and get the chatbot response."""
    chat_response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": user_input,
            },
        ]
    )
    return chat_response.choices[0].message.content

def eleven_labs_tts(text):
    """Convert text to speech using Eleven Labs API and get the audio URL."""
    api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_LABS_VOICE_ID}"
    headers = {
        'Content-Type': 'application/json',
        'xi-api-key': ELEVEN_LABS_API_KEY
    }
    payload = {"text": text}
    response = requests.post(api_url, json=payload, headers=headers)
    
    if response.status_code == 200:
        audio_url = response.json().get("audio_url")
        return audio_url
    else:
        print("Error generating speech")
        return None

def play_audio_from_url(audio_url):
    """Download and play the audio from the audio URL."""
    audio_response = requests.get(audio_url)
    if audio_response.status_code == 200:
        audio_file = BytesIO(audio_response.content)
        audio = AudioSegment.from_file(audio_file)
        play(audio)
    else:
        print("Failed to download audio")

def main():
    """Main loop for the Teddy Bear AI interaction."""
    print("Immy the AI Teddy Bear is ready to talk!")
    
    while True:
        user_input = recognize_speech()
        if user_input:
            response_text = send_to_Mistral(user_input)
            if response_text:
                print(f"Immy: {response_text}")
                audio_url = eleven_labs_tts(response_text)
                if audio_url:
                    play_audio_from_url(audio_url)
        time.sleep(1)  # Add a small delay to avoid rapid firing

if __name__ == "__main__":
    main()
