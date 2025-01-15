import os
import sys
import time
import queue
import threading
import pygame
import speech_recognition as sr
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from groq import Groq
from dotenv import load_dotenv
from pydub import AudioSegment

# Set the path to ffmpeg explicitly
ffmpeg_path = "C:\\ffmpeg\\bin\\ffmpeg.exe"  # Update this path to match your FFmpeg installation
AudioSegment.converter = ffmpeg_path

# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from environment variables
GROQ_API_KEY = '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'
ELEVENLABS_API_KEY = 'sk_dee83966a5e3d57289bb6ed748776fb374cac26e29f931a4'

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
eleven_labs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

class AudioStreamPlayer:
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=1)  # Initialize with specific settings
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.current_buffer = BytesIO()
        
    def add_audio_chunk(self, chunk):
        if chunk:
            self.audio_queue.put(chunk)
            
    def play_audio_stream(self):
        while True:
            if not self.is_playing and not self.audio_queue.empty():
                # Collect accumulated chunks
                self.current_buffer = BytesIO()
                while not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    self.current_buffer.write(chunk)
                
                self.current_buffer.seek(0)
                try:
                    # Load and play the WAV data
                    pygame.mixer.music.load(self.current_buffer, "wav")
                    pygame.mixer.music.play()
                    self.is_playing = True
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    self.is_playing = False
                except Exception as e:
                    print(f"Error playing audio: {e}")
                    self.is_playing = False
            time.sleep(0.1)

def stream_to_eleven_labs(text_queue: queue.Queue, audio_player: AudioStreamPlayer):
    accumulated_text = ""
    while True:
        while not text_queue.empty():
            text_chunk = text_queue.get()
            accumulated_text += text_chunk
            
            # Process text when we have enough for natural speech
            if len(accumulated_text.strip()) > 0 and (accumulated_text.strip()[-1] in '.!?'):
                try:
                    audio_stream = eleven_labs_client.text_to_speech.convert_as_stream(
                        voice_id="jBpfuIE2acCO8z3wKNLl",  # Adam pre-made voice
                        output_format="mp3_22050_32",
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
                    
                    # Save MP3 data to a temporary file
                    mp3_data = b"".join(audio_stream)
                    mp3_buffer = BytesIO(mp3_data)
                    
                    # Convert MP3 to WAV using pydub
                    audio = AudioSegment.from_file(mp3_buffer, format="mp3")
                    wav_buffer = BytesIO()
                    audio.export(wav_buffer, format="wav")
                    wav_buffer.seek(0)
                    
                    # Add WAV data to the audio player
                    audio_player.add_audio_chunk(wav_buffer.read())
                    
                    accumulated_text = ""  # Reset after processing
                    
                except Exception as e:
                    print(f"Error in text-to-speech conversion: {e}")
                
        time.sleep(0.1)

def send_to_groq_streaming(user_input: str, text_queue: queue.Queue) -> None:
    system_prompt = (
        "You are Immy, a magical AI-powered teddy bear who loves to chat with children. "
        "You are kind, funny, and full of wonder, always ready to tell stories, answer questions, and offer friendly advice. "
        "When speaking, you are playful, patient, and use simple, child-friendly language. You encourage curiosity, learning, and imagination."
        "keep your responses short and cute"
        "Dont use emojis in your responses. "
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

def listen_for_wake_word():
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            print("Waiting for wake word 'hi there'...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio).lower()
                if "hi there" in text:
                    print("Wake word detected!")
                    return True
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    text_queue = queue.Queue()
    audio_player = AudioStreamPlayer()
    
    # Start audio player thread
    audio_thread = threading.Thread(
        target=audio_player.play_audio_stream,
        daemon=True
    )
    audio_thread.start()
    
    # Start text-to-speech conversion thread
    tts_thread = threading.Thread(
        target=stream_to_eleven_labs,
        args=(text_queue, audio_player),
        daemon=True
    )
    tts_thread.start()
    
    while True:
        if listen_for_wake_word():
            user_input = recognize_speech()
            if user_input:
                # Start Groq streaming in a separate thread
                groq_thread = threading.Thread(
                    target=send_to_groq_streaming,
                    args=(user_input, text_queue),
                    daemon=True
                )
                groq_thread.start()

if __name__ == "__main__":
    main()