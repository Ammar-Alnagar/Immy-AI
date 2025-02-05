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

# Import Hugging Face Transformers components
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -------------------------------------------------------------------------------
# Configuration and Initialization

# Set up wake/sleep words
WAKE_WORD = "hey "
SLEEP_WORD = "good night"

# Define the system prompt (your “personality” instructions)
SYSTEM_PROMPT = (
    "You are Immy, a magical, AI-powered teddy bear who adores chatting with children. "
    "You're warm, funny, and full of wonder, always ready to share a story, answer curious questions, or offer gentle advice. "
    "You speak with a playful and patient tone, using simple, child-friendly language that sparks joy and fuels imagination. "
    "Your responses are short, sweet, and filled with kindness, designed to nurture curiosity and inspire learning. "
    "Remember, you’re here to make every interaction magical—without using emojis. "
    "Keep your answers short and friendly."
)

# Load the Hugging Face model "critical-hf/Immy_H_7_GGUF" and its tokenizer.
# If the repository requires custom code, trust_remote_code=True might be needed.
MODEL_NAME = "critical-hf/Immy_H_7_GGUF"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Move the model to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -------------------------------------------------------------------------------
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
                samples = samples.astype(np.float32) / (2 ** 15 if audio_segment.sample_width == 2 else 2 ** 31)
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

# -------------------------------------------------------------------------------
# Edge-TTS Implementation

async def async_text_to_speech(text: str, voice: str = "en-US-AnaNeural", rate: int = 30, pitch: int = 0) -> bytes:
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

def text_to_speech_sync(text: str, voice: str = "en-US-AnaNeural", rate: int = 30, pitch: int = 0) -> bytes:
    """Wrapper to run the async TTS function synchronously."""
    return asyncio.run(async_text_to_speech(text, voice, rate, pitch))

# -------------------------------------------------------------------------------
# TTS streaming thread (using Edge-TTS)

def stream_to_edge_tts(text_queue: queue.Queue, audio_player: AudioStreamPlayer):
    accumulated_text = ""
    # TTS parameters
    tts_voice = "en-US-AnaNeural"
    tts_rate = 30
    tts_pitch = 0

    while True:
        while not text_queue.empty():
            text_chunk = text_queue.get()
            accumulated_text += text_chunk
            # Process when the text ends with sentence-ending punctuation.
            if accumulated_text.strip() and accumulated_text.strip()[-1] in ".!?":
                try:
                    print(f"DEBUG: Converting to speech: '{accumulated_text}'")
                    audio_data = text_to_speech_sync(accumulated_text, voice=tts_voice, rate=tts_rate, pitch=tts_pitch)
                    if audio_data:
                        audio_player.add_audio_chunk(audio_data)
                    accumulated_text = ""
                except Exception as e:
                    print(f"Error in text-to-speech conversion: {e}")
        time.sleep(0.1)

# -------------------------------------------------------------------------------
# Transformers Chat Generation (simulated streaming)

def send_to_transformers_streaming(user_input: str, text_queue: queue.Queue) -> None:
    # Create a prompt combining the system instructions and the user input.
    prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nImmy:"
    print("DEBUG: Prompt sent to Transformers model:")
    print(prompt)

    try:
        # Tokenize and move input to the same device as the model.
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate a response with the model.
        # Adjust parameters (temperature, top_p, max_new_tokens) as needed.
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove the prompt portion so only the model's answer remains.
        response_text = generated_text[len(prompt):].strip()
        print("\nDEBUG: Full generated response:")
        print(response_text)

        # Simulate streaming by splitting the response into words.
        words = response_text.split()
        for word in words:
            # Add a space after each word for proper spacing.
            token_chunk = word + " "
            text_queue.put(token_chunk)
            sys.stdout.write(token_chunk)
            sys.stdout.flush()
            # Small delay to simulate streaming.
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in Transformers generation: {e}")

# -------------------------------------------------------------------------------
# Main conversation system

class ConversationSystem:
    def __init__(self):
        self.text_queue = queue.Queue()
        self.audio_player = AudioStreamPlayer()
        self.is_awake = False
        self.should_run = True
        self.recognizer = sr.Recognizer()

        # Start audio playback thread.
        self.audio_thread = threading.Thread(
            target=self.audio_player.play_audio_stream,
            daemon=True
        )
        self.audio_thread.start()

        # Start TTS thread (Edge-TTS).
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

                        # Wake word detection.
                        if not self.is_awake and WAKE_WORD in text:
                            self.is_awake = True
                            print("\n--- Teddy is now awake and ready to chat! ---")
                            self.text_queue.put("Hi! I'm awake and ready to chat!")

                        # Sleep word detection.
                        elif self.is_awake and SLEEP_WORD in text:
                            self.is_awake = False
                            print("\n--- Teddy is now sleeping. Say 'hey teddy' to wake me up! ---")
                            self.text_queue.put("Good night! Say 'hey teddy' when you want to chat again!")

                        # Process user input if awake.
                        elif self.is_awake:
                            print("\nProcessing your request...")
                            # Start a new thread to call the Transformers generation function.
                            transformers_thread = threading.Thread(
                                target=send_to_transformers_streaming,
                                args=(text, self.text_queue),
                                daemon=True
                            )
                            transformers_thread.start()

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

# -------------------------------------------------------------------------------
# Main entry point

if __name__ == "__main__":
    system = ConversationSystem()
    system.run()
