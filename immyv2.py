import os
import sys
import time
import queue
import threading
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from typing import Iterator
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from groq import Groq
from dotenv import load_dotenv
import numpy as np



# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = 'sk_dee83966a5e3d57289bb6ed748776fb374cac26e29f931a4'

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
eleven_labs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Define wake and sleep words
WAKE_WORD = "hey teddy"
SLEEP_WORD = "good night teddy"

class AudioStreamPlayer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False
        
    def add_audio_chunk(self, chunk):
        if chunk:
            self.audio_queue.put(chunk)
    
    def play_audio_file(self, audio_data):
        try:
            with BytesIO(audio_data) as audio_file:
                data, samplerate = sf.read(audio_file)
                sd.play(data, samplerate)
                sd.wait()
        except Exception as e:
            print(f"Error playing audio: {e}")
            
    def play_audio_stream(self):
        while True:
            if not self.is_playing and not self.audio_queue.empty():
                try:
                    self.is_playing = True
                    audio_data = BytesIO()
                    
                    # Collect all available chunks
                    while not self.audio_queue.empty():
                        chunk = self.audio_queue.get()
                        audio_data.write(chunk)
                    
                    audio_data.seek(0)
                    self.play_audio_file(audio_data.getvalue())
                    self.is_playing = False
                except Exception as e:
                    print(f"Error in audio playback: {e}")
                    self.is_playing = False
            time.sleep(0.1)

def stream_to_eleven_labs(text_queue: queue.Queue, audio_player: AudioStreamPlayer):
    accumulated_text = ""
    while True:
        while not text_queue.empty():
            text_chunk = text_queue.get()
            accumulated_text += text_chunk
            
            if len(accumulated_text.strip()) > 0 and (accumulated_text.strip()[-1] in '.!?'):
                try:
                    audio_stream = eleven_labs_client.text_to_speech.convert_as_stream(
                        voice_id="jBpfuIE2acCO8z3wKNLl",  # Adam pre-made voice
                        output_format="mp3_44100_128",  # Changed format for better compatibility
                        optimize_streaming_latency="4",
                        text=accumulated_text,
                        model_id="eleven_turbo_v2_5",
                        voice_settings=VoiceSettings(
                            stability=0.0,
                            similarity_boost=1.0,
                            style=0.0,
                            use_speaker_boost=True,
                        ),
                    )
                    
                    for audio_chunk in audio_stream:
                        audio_player.add_audio_chunk(audio_chunk)
                    
                    accumulated_text = ""
                    
                except Exception as e:
                    print(f"Error in text-to-speech conversion: {e}")
                
        time.sleep(0.1)

def send_to_groq_streaming(user_input: str, text_queue: queue.Queue) -> None:
    system_prompt = (
        "Your name is Immy, a magical AI-powered teddy bear who loves to chat with children. "
        "You are kind, funny, and full of wonder, always ready to tell stories, answer questions, and offer friendly advice. "
        "When speaking, you are playful, patient, and use simple, child-friendly language. You encourage curiosity, learning, and imagination. "
        "keep your responses short and cute. "
        "Don't use emojis in your responses. "
    )
    
    try:
        stream = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                text_queue.put(content)
                sys.stdout.write(content)
                sys.stdout.flush()
                
    except Exception as e:
        print(f"Error in Groq API call: {e}")

class ConversationSystem:
    def __init__(self):
        self.text_queue = queue.Queue()
        self.audio_player = AudioStreamPlayer()
        self.is_awake = False
        self.should_run = True
        self.recognizer = sr.Recognizer()
        
        # Start audio player thread
        self.audio_thread = threading.Thread(
            target=self.audio_player.play_audio_stream,
            daemon=True
        )
        self.audio_thread.start()
        
        # Start text-to-speech conversion thread
        self.tts_thread = threading.Thread(
            target=stream_to_eleven_labs,
            args=(self.text_queue, self.audio_player),
            daemon=True
        )
        self.tts_thread.start()

    def listen_continuously(self):
        with sr.Microphone() as source:
            print("\nListening...")
            
            while self.should_run:
                try:
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"\nRecognized: {text}")
                        
                        if not self.is_awake and WAKE_WORD in text:
                            self.is_awake = True
                            print("\n--- Teddy is now awake and ready to chat! ---")
                            self.text_queue.put("Hi! I'm awake and ready to chat!")
                        elif self.is_awake and SLEEP_WORD in text:
                            self.is_awake = False
                            print("\n--- Teddy is now sleeping. Say 'hey teddy' to wake me up! ---")
                            self.text_queue.put("Good night! Say 'hey teddy' when you want to chat again!")
                        elif self.is_awake:
                            groq_thread = threading.Thread(
                                target=send_to_groq_streaming,
                                args=(text, self.text_queue),
                                daemon=True
                            )
                            groq_thread.start()
                            # Wait for the response to finish before listening again
                            groq_thread.join()
                            print("\nListening...")
                            
                    except sr.UnknownValueError:
                        if self.is_awake:
                            print("\nListening...")
                        continue
                    except sr.RequestError as e:
                        print(f"Could not request results: {e}")
                        continue
                        
                except KeyboardInterrupt:
                    self.should_run = False
                    break

    def run(self):
        print("\nWelcome to Teddy Bear Chat!")
        print(f"Say '{WAKE_WORD}' to wake me up")
        print(f"Say '{SLEEP_WORD}' to put me to sleep")
        print("Press Ctrl+C to exit")
        
        try:
            self.listen_continuously()
        except KeyboardInterrupt:
            print("\nGoodbye! Thanks for chatting!")
            self.should_run = False

if __name__ == "__main__":
    system = ConversationSystem()
    system.run()