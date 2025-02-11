import os
import sys
import time
import queue
import threading
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from io import BytesIO
import numpy as np
import asyncio
import tempfile
import edge_tts
import whisper  # <-- Import Whisper for speech recognition

# Import Ollama's Python client (make sure you have it installed, e.g., via `pip install ollama`)
import ollama

# ------------------------------------------------------------------------------
# Configuration and Initialization

# Set up wake/sleep words
WAKE_WORD = "hey "
SLEEP_WORD = "good night"

# Set the model name to be used by Ollama (adjust as needed)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "immy_hermes_v2")

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
# Audio Playback Class

class AudioStreamPlayer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False

    def add_audio_chunk(self, chunk):
        if chunk:
            self.audio_queue.put(chunk)

    def play_audio_file(self, audio_data):
        try:
            # Use pydub to load the MP3 data.
            from pydub import AudioSegment
            with BytesIO(audio_data) as audio_buffer:
                audio_segment = AudioSegment.from_mp3(audio_buffer)
                # Convert to numpy array.
                samples = np.array(audio_segment.get_array_of_samples())
                # Normalize samples to float32.
                samples = samples.astype(np.float32) / (2 ** 15 if audio_segment.sample_width == 2 else 2 ** 31)
                # If mono, duplicate channel for stereo playback.
                if audio_segment.channels == 1:
                    samples = np.column_stack((samples, samples))
                # Play audio using sounddevice.
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
                    # Combine all queued audio chunks.
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
# Edge-TTS Implementation (for text-to-speech conversion)
#
# This asynchronous function uses Edge-TTS to synthesize speech into an MP3 file,
# then reads the file's bytes so it can be played by our audio player.

async def async_text_to_speech(text: str, voice: str = "en-US-AnaNeural", rate: int = 25, pitch: int = 0) -> bytes:
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


def text_to_speech_sync(text: str, voice: str = "en-US-AnaNeural", rate: int = 25, pitch: int = 0) -> bytes:
    """Wrapper to run the async TTS function synchronously."""
    return asyncio.run(async_text_to_speech(text, voice, rate, pitch))


# ------------------------------------------------------------------------------
# TTS Streaming Thread (Edge-TTS)
#
# This function listens to a text queue and, once a sentence (ending with punctuation)
# is accumulated, converts the text into speech and queues the audio for playback.

def stream_to_edge_tts(text_queue: queue.Queue, audio_player: AudioStreamPlayer):
    accumulated_text = ""
    tts_voice = "en-US-AnaNeural"
    tts_rate = 25
    tts_pitch = 0

    while True:
        while not text_queue.empty():
            text_chunk = text_queue.get()
            accumulated_text += text_chunk
            # Process once we have a complete sentence.
            if accumulated_text.strip() and accumulated_text.strip()[-1] in ".!?":
                try:
                    audio_data = text_to_speech_sync(accumulated_text, voice=tts_voice, rate=tts_rate, pitch=tts_pitch)
                    if audio_data:
                        audio_player.add_audio_chunk(audio_data)
                    accumulated_text = ""
                except Exception as e:
                    print(f"Error in text-to-speech conversion: {e}")
        time.sleep(0.1)


# ------------------------------------------------------------------------------
# Ollama Chat Generation
#
# This function sends a prompt (which includes the system instructions and the user input)
# to the Ollama model with streaming enabled. Here we pass the model name and prompt
# as positional arguments.
def send_to_ollama_streaming(user_input: str, text_queue: queue.Queue) -> None:
    prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nImmy:"
    try:
        # Call the Ollama chat API.
        # Pass the model name and the prompt as positional arguments.
        for token in ollama.chat(OLLAMA_MODEL, prompt, stream=True):
            if token:
                text_queue.put(token)
                sys.stdout.write(token)
                sys.stdout.flush()
    except Exception as e:
        print(f"Error in Ollama generation: {e}")


# ------------------------------------------------------------------------------
# Main Conversation System

class ConversationSystem:
    def __init__(self):
        self.text_queue = queue.Queue()
        self.audio_player = AudioStreamPlayer()
        self.is_awake = False
        self.should_run = True
        self.recognizer = sr.Recognizer()

        # Load the Whisper model once (using the "large-v3-turbo" variant)
        print("Loading Whisper model (this may take a while)...")
        self.whisper_model = whisper.load_model("large-v3-turbo")
        print("Whisper model loaded.")

        # Start the audio playback thread.
        self.audio_thread = threading.Thread(
            target=self.audio_player.play_audio_stream,
            daemon=True
        )
        self.audio_thread.start()

        # Start the TTS thread.
        self.tts_thread = threading.Thread(
            target=stream_to_edge_tts,
            args=(self.text_queue, self.audio_player),
            daemon=True
        )
        self.tts_thread.start()

    def transcribe_with_whisper(self, audio: sr.AudioData) -> str:
        """
        Transcribe audio using OpenAI's Whisper model.
        The audio is first written to a temporary WAV file,
        then processed by Whisper.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio.get_wav_data())
                tmp_filename = tmp_file.name
            result = self.whisper_model.transcribe(tmp_filename)
            os.remove(tmp_filename)
            return result["text"]
        except Exception as e:
            print(f"Error transcribing audio with Whisper: {e}")
            return ""

    def listen_continuously(self):
        with sr.Microphone() as source:
            print("\nListening...")
            # Adjust for ambient noise.
            self.recognizer.adjust_for_ambient_noise(source)

            while self.should_run:
                try:
                    print("\nSay something...")
                    # Capture audio for a phrase (limit to 5 seconds)
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
                    try:
                        # Transcribe using Whisper instead of Google Speech Recognition.
                        text = self.transcribe_with_whisper(audio).lower()
                        print(f"\nRecognized: {text}")

                        # Detect wake word.
                        if not self.is_awake and WAKE_WORD in text:
                            self.is_awake = True
                            print("\n--- Teddy is now awake and ready to chat! ---")
                            self.text_queue.put("Hi! I'm awake and ready to chat!")

                        # Detect sleep word.
                        elif self.is_awake and SLEEP_WORD in text:
                            self.is_awake = False
                            print("\n--- Teddy is now sleeping. Say 'hey teddy' to wake me up! ---")
                            self.text_queue.put("Good night! Say 'hey teddy' when you want to chat again!")

                        # Process user input if awake.
                        elif self.is_awake:
                            print("\nProcessing your request...")
                            ollama_thread = threading.Thread(
                                target=send_to_ollama_streaming,
                                args=(text, self.text_queue),
                                daemon=True
                            )
                            ollama_thread.start()

                    except Exception as e:
                        print(f"\nError during transcription: {e}")
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
# Main Entry Point

if __name__ == "__main__":
    system = ConversationSystem()
    system.run()
