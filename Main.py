import speech_recognition as sr
import requests
import io
from pydub import AudioSegment
from pydub.playback import play
import time

# Eleven Labs API settings


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

def send_to_LLMinBox(user_input):
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {"text": user_input}

    response = requests.post(LLMINABOX_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()['response']
    else:
        print("Failed to get response from LLMinaBox")
        return None

def eleven_labs_tts(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_LABS_VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_LABS_API_KEY
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
    print("Immy the AI Teddy Bear is ready!")
    while True:
        user_input = recognize_speech()
        if user_input:
            llminabox_response = send_to_LLMinBox(user_input)
            if llminabox_response:
                print(f"Immy: {llminabox_response}")
                audio_content = eleven_labs_tts(llminabox_response)
                if audio_content:
                    play_audio(audio_content)
        time.sleep(1)  # Small delay to prevent continuous listening

if __name__ == "__main__":
    main()