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
import numpy as np
import asyncio
import tempfile
import edge_tts

# Import llama-cpp for local LLaMA model inference
from llama_cpp import Llama

# ------------------------------------------------------------------------------
# Configuration and Initialization

# Set up wake/sleep words
WAKE_WORD = "hey "
SLEEP_WORD = "good night"

# Set the path to your GGUF model file (adjust as needed)
LLAMA_MODEL_PATH = os.getenv("models", "models/immy_hermes_v2-q4_k_m.gguf")

# Initialize the LLaMA model
llama_model = Llama(
    model_path=LLAMA_MODEL_PATH,
    n_ctx=1024,       # adjust context length if needed
    seed=0,
    verbose=False,
    n_threads=8
)

# Define the system prompt (your “personality” instructions)
SYSTEM_PROMPT = (
    "You are Immy, a magical, AI-powered teddy bear who adores chatting with children. "
    "You're warm, funny, and full of wonder, always ready to share a story, answer curious questions, or offer gentle advice. "
    "You speak with a playful and patient tone, using simple, child-friendly language that sparks joy and fuels imagination. "
    "Your responses are short, sweet, and filled with kindness, designed to nurture curiosity and inspire learning. "
    "Remember, you’re here to make every interaction magical—without using emojis. "
    "Keep your answers short and friendly."
)

# ------------------------------------------------------------------------------
# Audio playback class

class AudioStreamPlayer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False
        
    def add_audio_chunk(self, chunk):
        if chunk:
            self.audio_queue.put(chunk)
    
    def play_audio_file(self, audio_data):
        try:
            # Load audio with pydub (handles MP3)
            from pydub import AudioSegment
            with BytesIO(audio_data) as audio_buffer:
                audio_segment = AudioSegment.from_mp3(audio_buffer)
                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples())
                # Normalize samples to float32
                samples = samples.astype(np.float32) / (2**15 if audio_segment.sample_width == 2 else 2**31)
                # If mono, duplicate channel for stereo playback
                if audio_segment.channels == 1:
                    samples = np.column_stack((samples, samples))
                # Play audio using sounddevice
                sd.play(samples, audio_segment.frame_rate)
                sd.wait()
        except Exception as e:
            print(f"Error playing audio: {e}")
            
    def play_audio_stream(self):
        while True:
            if not self.is_playing and not self.audio_queue.empty():
                try:
                    self.is_playing = True
                    audio_data = BytesIO()
                    # Combine all queued audio chunks
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

# ------------------------------------------------------------------------------
# Edge-TTS Implementation (replacing ElevenLabs)
#
# This asynchronous function uses Edge-TTS to synthesize speech to an MP3 file,
# then reads back the file's bytes so it can be played by our audio player.
# You can adjust the default voice, rate, and pitch as desired.

async def async_text_to_speech(text: str, voice: str = "en-US-AnaNeural", rate: int = 0, pitch: int = 0) -> bytes:
    if not text.strip():
        return None
    rate_str = f"{rate:+d}%"
    pitch_str = f"{pitch:+d}Hz"
    communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
    await communicate.save(tmp_path)
    with open(tmp_path, "rb") as f:
        audio_data = f.read()
    os.remove(tmp_path)
    return audio_data

def text_to_speech_sync(text: str, voice: str = "en-US-AnaNeural", rate: int = 0, pitch: int = 0) -> bytes:
    """Wrapper to run the async TTS function synchronously."""
    return asyncio.run(async_text_to_speech(text, voice, rate, pitch))

# ------------------------------------------------------------------------------
# TTS streaming thread (replacing ElevenLabs streaming)

def stream_to_edge_tts(text_queue: queue.Queue, audio_player: AudioStreamPlayer):
    accumulated_text = ""
    # You may adjust these TTS parameters as desired.
    tts_voice = "en-US-AnaNeural"
    tts_rate = 0
    tts_pitch = 0

    while True:
        while not text_queue.empty():
            text_chunk = text_queue.get()
            accumulated_text += text_chunk
            # When the accumulated text ends with a sentence-ending punctuation, process it.
            if accumulated_text.strip() and accumulated_text.strip()[-1] in ".!?":
                try:
                    # Convert the accumulated text to speech (synchronously calling our async wrapper)
                    audio_data = text_to_speech_sync(accumulated_text, voice=tts_voice, rate=tts_rate, pitch=tts_pitch)
                    if audio_data:
                        audio_player.add_audio_chunk(audio_data)
                    accumulated_text = ""
                except Exception as e:
                    print(f"Error in text-to-speech conversion: {e}")
        time.sleep(0.1)

# ------------------------------------------------------------------------------
# LLaMA Chat Generation (replacing OpenAI ChatGPT)

def send_to_llama_streaming(user_input: str, text_queue: queue.Queue) -> None:
    # Create a prompt combining the system instructions and the user input.
    prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nImmy:"
    
    try:
        # Call the LLaMA model in streaming mode.
        # The llama_cpp package returns a generator yielding tokens.
        response = llama_model(prompt=prompt, stream=True)
        for token in response:
            # Depending on your llama_cpp version the token may be a dict or a string.
            # Here we assume a dict with key 'token' for simplicity.
            token_text = token.get("token", "")
            if token_text:
                text_queue.put(token_text)
                sys.stdout.write(token_text)
                sys.stdout.flush()
    except Exception as e:
        print(f"Error in LLaMA generation: {e}")

# ------------------------------------------------------------------------------
# Main conversation system

class ConversationSystem:
    def __init__(self):
        self.text_queue = queue.Queue()
        self.audio_player = AudioStreamPlayer()
        self.is_awake = False
        self.should_run = True
        self.recognizer = sr.Recognizer()
        
        # Start audio playback thread
        self.audio_thread = threading.Thread(
            target=self.audio_player.play_audio_stream,
            daemon=True
        )
        self.audio_thread.start()
        
        # Start TTS thread (Edge-TTS)
        self.tts_thread = threading.Thread(
            target=stream_to_edge_tts,
            args=(self.text_queue, self.audio_player),
            daemon=True
        )
        self.tts_thread.start()

    def listen_continuously(self):
        with sr.Microphone() as source:
            print("\nListening...")
            # Adjust for ambient noise to improve recognition accuracy.
            self.recognizer.adjust_for_ambient_noise(source)
            
            while self.should_run:
                try:
                    print("\nSay something...")
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"\nRecognized: {text}")
                        
                        # Detect wake word to “wake” the teddy.
                        if not self.is_awake and WAKE_WORD in text:
                            self.is_awake = True
                            print("\n--- Teddy is now awake and ready to chat! ---")
                            self.text_queue.put("Hi! I'm awake and ready to chat!")
                        
                        # Detect sleep word to put the teddy to sleep.
                        elif self.is_awake and SLEEP_WORD in text:
                            self.is_awake = False
                            print("\n--- Teddy is now sleeping. Say 'hey teddy' to wake me up! ---")
                            self.text_queue.put("Good night! Say 'hey teddy' when you want to chat again!")
                        
                        # Process user input if awake.
                        elif self.is_awake:
                            print("\nProcessing your request...")
                            # Start a new thread to call the LLaMA generation function.
                            llama_thread = threading.Thread(
                                target=send_to_llama_streaming,
                                args=(text, self.text_queue),
                                daemon=True
                            )
                            llama_thread.start()
                            
                    except sr.UnknownValueError:
                        print("\nCould not understand audio.")
                        continue
                    except sr.RequestError as e:
                        print(f"\nCould not request results from Google Speech Recognition service: {e}")
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

# ------------------------------------------------------------------------------
# Main entry point

if __name__ == "__main__":
    system = ConversationSystem()
    system.run()
