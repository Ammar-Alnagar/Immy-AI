import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
import time
import os

# Optional: Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

WAKE_WORD = "hi"
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
    # Simulate a Groq API response by returning a friendly response.
    # Replace this with the real Groq API call when needed.
    return f"Hello! You said: {user_input}. How can I help you?"


def tts_pydub(text):
    """Convert text to speech using gTTS and play the audio with pydub."""
    try:
        # Use gTTS to convert the text to speech
        tts = gTTS(text=text, lang='en')
        # Save the audio to a temporary file
        tts.save("response.mp3")

        # Use pydub to play the saved audio
        audio = AudioSegment.from_mp3("response.mp3")
        play(audio)

    except Exception as e:
        print(f"Error during TTS: {str(e)}")


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
                            tts_pydub("Goodbye!")
                            break
                        else:
                            groq_response = send_to_groq(user_input)
                            if groq_response:
                                print(f"Immy: {groq_response}")
                                tts_pydub(groq_response)
                    except Exception as e:
                        print(f"Error: {str(e)}")
                        continue
                
        time.sleep(1)  # Small delay before starting over

if __name__ == "__main__":
    main()
