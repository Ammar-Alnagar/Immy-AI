import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
import time
import os
import requests

# Load environment variables for API keys
from dotenv import load_dotenv
load_dotenv()

# Define wake-up and goodbye words
WAKE_WORD = "hey"
GOODBYE_WORD = "goodbye"

# Load API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define the Groq API endpoint
GROQ_API_URL = "https://api.groq.com/v1/completions"

def recognize_speech():
    """
    Continuously listens for the wake word.
    When the wake word is detected, returns the recognized speech.
    """
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
    """
    Sends user input to the Groq API and returns the generated response.
    """
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",  # Adjust model version if needed
        "messages": [
            {"role": "system", "content": "You are Immy, an AI teddy bear designed to interact with children. Respond in a friendly, simple, and engaging manner."},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,  # Adjust as needed
        "max_tokens": 150  # Adjust as needed
    }

    try:
        # Make the API request
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Check for HTTP errors
        
        # Parse the response from the API
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    return "Sorry, I couldn't process that request."


def tts_pydub(text):
    """
    Converts the given text to speech using gTTS and plays the audio using pydub.
    """
    try:
        # Use gTTS to convert the text to speech
        tts = gTTS(text=text, lang='en')
        # Save the audio to a temporary file
        tts.save("response.mp3")

        # Use pydub to load and play the saved audio
        audio = AudioSegment.from_mp3("response.mp3")
        play(audio)

    except Exception as e:
        print(f"Error during TTS: {str(e)}")


def main():
    """
    Main function to run the AI Teddy Bear.
    Listens for wake-up word, processes commands, and responds with TTS.
    """
    print("Immy the AI Teddy Bear is ready! Say 'Hey Immy' to wake up.")
    
    while True:
        # Listen for the wake-up word
        wake_input = recognize_speech()
        
        if WAKE_WORD in wake_input:
            print("Immy is now active!")
            while True:
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    print("Listening for commands...")
                    audio = recognizer.listen(source)
                    try:
                        # Recognize the user's speech
                        user_input = recognizer.recognize_google(audio).lower()
                        
                        if GOODBYE_WORD in user_input:
                            # Stop the conversation if the goodbye word is detected
                            print("Goodbye detected. Stopping the interaction.")
                            tts_pydub("Goodbye!")
                            break
                        else:
                            # Get a response from the Groq API and play it as speech
                            groq_response = send_to_groq(user_input)
                            if groq_response:
                                print(f"Immy: {groq_response}")
                                tts_pydub(groq_response)
                    except Exception as e:
                        print(f"Error: {str(e)}")
                        continue
                
        # Small delay before restarting the loop
        time.sleep(1)

if __name__ == "__main__":
    main()
