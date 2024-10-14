import speech_recognition as sr
import requests
import io
from pydub import AudioSegment
from pydub.playback import play
import time
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API")
os.environ["ELEVEN_LABS_API_KEY"] = os.getenv("ELEVEN_LABS_API_KEY")

WAKE_WORD = "hey immy"
GOODBYE_WORD = "goodbye"


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for wake word...")
        while True:
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio).lower()
                print(f"Recognized: {text}")
                if WAKE_WORD in text:
                    print("Wake word detected!")
                    return text
            except Exception as e:
                print(f"Error: {str(e)}")
                continue


def send_to_groq(user_input):
    headers = {
        'Authorization': f'Bearer {os.getenv("GROQ_API_KEY")}',
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are Immy, an AI teddy bear designed to interact with children. Respond in a friendly, simple, and engaging manner."},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
        "max_tokens": 750
    }
    response = requests.post("https://api.groq.com/v1/completions", json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Failed to get response from Groq: {response.status_code}")
        return None


def eleven_labs_tts(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{os.getenv('ELEVEN_LABS_VOICE_ID')}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": os.getenv("ELEVEN_LABS_API_KEY")
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return io.BytesIO(response.content)
    else:
        print("Failed to generate speech from Eleven Labs")
        return None


def play_audio(audio_content):
    audio = AudioSegment.from_mp3(audio_content)
    play(audio)


def main():
    print("Immy the AI Teddy Bear is ready! Say 'Hey Immy' to wake up.")
    
    while True:
        wake_input = recognize_speech()  # Listen for wake-up word
        
        if WAKE_WORD in wake_input:
            print("Immy is now active!")
            while True:
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    print("Listening for commands...")
                    audio = recognizer.listen(source)
                    try:
                        user_input = recognizer.recognize_google(audio).lower()
                        if GOODBYE_WORD in user_input:
                            print("Goodbye detected. Stopping the interaction.")
                            play_audio(eleven_labs_tts("Goodbye!"))
                            break
                        else:
                            groq_response = send_to_groq(user_input)
                            if groq_response:
                                print(f"Immy: {groq_response}")
                                audio_content = eleven_labs_tts(groq_response)
                                if audio_content:
                                    play_audio(audio_content)
                    except Exception as e:
                        print(f"Error: {str(e)}")
                        continue
                
        time.sleep(1)  # Small delay before starting over

if __name__ == "__main__":
    main()
