import os
import sys
import time
import queue
import threading
import sounddevice as sd
import speech_recognition as sr
from io import BytesIO
from typing import Optional
import numpy as np
import asyncio
import tempfile

# For audio file handling.
import soundfile as sf
from pydub import AudioSegment

# For environment variables (used for OpenAI API key)
from dotenv import load_dotenv
load_dotenv()

# For conversation generation with OpenAI
from openai import OpenAI

# For Edge-TTS (text-to-speech)
import edge_tts

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Define wake and sleep words
WAKE_WORD = "hey "
SLEEP_WORD = "good night"


class AudioStreamPlayer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_finished_callback: Optional[callable] = None  # Callback after playback finishes

    def add_audio_chunk(self, chunk):
        if chunk:
            self.audio_queue.put(chunk)

    def play_audio_file(self, audio_data):
        try:
            # Load the audio data using pydub from an in‑memory buffer.
            with BytesIO(audio_data) as audio_buffer:
                audio_segment = AudioSegment.from_mp3(audio_buffer)
                # Convert to numpy array.
                samples = np.array(audio_segment.get_array_of_samples())
                # Normalize to float32.
                samples = samples.astype(np.float32) / (2 ** 15 if audio_segment.sample_width == 2 else 2 ** 31)
                # If mono, convert to stereo.
                if audio_segment.channels == 1:
                    samples = np.column_stack((samples, samples))
                # Play the audio.
                self.is_playing = True
                sd.play(samples, audio_segment.frame_rate)
                sd.wait()
                self.is_playing = False

                # Call the playback finished callback if set.
                if self.playback_finished_callback:
                    self.playback_finished_callback()

        except Exception as e:
            print(f"Error playing audio: {e}")
            self.is_playing = False

    def play_audio_stream(self):
        while True:
            if not self.is_playing and not self.audio_queue.empty():
                try:
                    audio_data = BytesIO()
                    # Combine all available chunks.
                    while not self.audio_queue.empty():
                        chunk = self.audio_queue.get()
                        audio_data.write(chunk)
                    audio_data.seek(0)
                    self.play_audio_file(audio_data.getvalue())
                except Exception as e:
                    print(f"Error in audio playback: {e}")
                    self.is_playing = False
            time.sleep(0.1)


# -------------------------------
# Edge-TTS Conversion Functions
# -------------------------------
async def async_text_to_speech(text: str, voice: str = "en-US-AnaNeural", rate: int = 25, pitch: int = 10) -> bytes:
    if not text.strip():
        return None
    # Prepare rate and pitch strings (e.g. "+25%" and "+10Hz")
    rate_str = f"{rate:+d}%"
    pitch_str = f"{pitch:+d}Hz"
    communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)
    # Save the audio to a temporary MP3 file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
    await communicate.save(tmp_path)
    # Read the file data.
    with open(tmp_path, "rb") as f:
        audio_data = f.read()
    os.remove(tmp_path)
    return audio_data

def text_to_speech_sync(text: str, voice: str = "en-US-AnaNeural", rate: int = 25, pitch: int = 10) -> bytes:
    """Synchronous wrapper to run the async Edge-TTS conversion."""
    return asyncio.run(async_text_to_speech(text, voice, rate, pitch))


# ----------------------------------------------
# Edge-TTS Streaming Thread (replaces ElevenLabs)
# ----------------------------------------------
def stream_to_edge_tts(text_queue: queue.Queue, audio_player: AudioStreamPlayer):
    accumulated_text = ""
    tts_voice = "en-US-AnaNeural"  # You can change this to your preferred voice.
    tts_rate = 25
    tts_pitch = 10

    while True:
        # Wait if the audio player is busy to prevent overlapping playback.
        if audio_player.is_playing:
            time.sleep(0.1)
            continue

        while not text_queue.empty():
            text_chunk = text_queue.get()
            if not isinstance(text_chunk, str):
                text_chunk = str(text_chunk)
            accumulated_text += text_chunk
            # When the accumulated text ends with sentence-ending punctuation, perform TTS.
            if accumulated_text.strip() and accumulated_text.strip()[-1] in ".!?":
                try:
                    audio_data = text_to_speech_sync(accumulated_text, voice=tts_voice, rate=tts_rate, pitch=tts_pitch)
                    if audio_data:
                        audio_player.add_audio_chunk(audio_data)
                    accumulated_text = ""
                except Exception as e:
                    print(f"Error in text-to-speech conversion: {e}")
        time.sleep(0.1)


# ---------------------------------------------
# OpenAI Chat Generation (streaming response)
# ---------------------------------------------
def send_to_openai_streaming(user_input: str, text_queue: queue.Queue,
                             conversation_history: list, history_lock: threading.Lock) -> None:
    system_prompt = (
        "You are Immy, a magical, AI-powered teddy bear who adores chatting with children. "
        "You're warm, funny, and full of wonder, always ready to share a story, answer curious questions, or offer gentle advice. "
        "You speak with a playful and patient tone, using simple, child-friendly language that sparks joy and fuels imagination. "
        "Your responses are short, sweet, and filled with kindness, designed to nurture curiosity and inspire learning. "
        "Remember, you’re here to make every interaction magical—without using emojis. "
        "Keep your answers short and friendly."
    )

    with history_lock:
        conversation_history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": system_prompt}] + conversation_history.copy()

    try:
        stream = openai_client.chat.completions.create(
            model="chatgpt-4o-latest",  # Adjust model as needed.
            messages=messages,
            stream=True
        )

        answer_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                answer_text += content
                text_queue.put(content)
                sys.stdout.write(content)
                sys.stdout.flush()

        with history_lock:
            conversation_history.append({"role": "assistant", "content": answer_text})

    except Exception as e:
        print(f"Error in OpenAI API call: {e}")


# ------------------------------
# Conversation System Class
# ------------------------------
class ConversationSystem:
    def __init__(self):
        self.text_queue = queue.Queue()
        self.audio_player = AudioStreamPlayer()
        self.is_awake = False
        self.should_run = True
        self.recognizer = sr.Recognizer()

        # For conversation context
        self.conversation_history = []
        self.history_lock = threading.Lock()

        # Timestamp for when the last TTS utterance finished playing
        self.last_tts_end = 0.0
        # Minimum delay (in seconds) after TTS playback before processing microphone input
        self.post_tts_cooldown = 1.0

        # Set the callback so that when audio playback finishes, we update last_tts_end.
        self.audio_player.playback_finished_callback = self.audio_finished_callback

        # Start audio player thread
        self.audio_thread = threading.Thread(
            target=self.audio_player.play_audio_stream,
            daemon=True
        )
        self.audio_thread.start()

        # Start text-to-speech conversion thread using Edge-TTS.
        self.tts_thread = threading.Thread(
            target=stream_to_edge_tts,
            args=(self.text_queue, self.audio_player),
            daemon=True
        )
        self.tts_thread.start()

    def audio_finished_callback(self):
        self.last_tts_end = time.time()

    def listen_continuously(self):
        with sr.Microphone() as source:
            print("\nListening...")
            self.recognizer.adjust_for_ambient_noise(source)

            while self.should_run:
                if self.audio_player.is_playing:
                    time.sleep(0.1)
                    continue

                if time.time() - self.last_tts_end < self.post_tts_cooldown:
                    time.sleep(0.1)
                    continue

                try:
                    print("\nSay something...")
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"\nRecognized: {text}")

                        # Wake word detection
                        if not self.is_awake and WAKE_WORD in text:
                            self.is_awake = True
                            print("\n--- Teddy is now awake and ready to chat! ---")
                            self.text_queue.put("Hi! I'm awake and ready to chat!")

                        # Sleep word detection
                        elif self.is_awake and SLEEP_WORD in text:
                            self.is_awake = False
                            print("\n--- Teddy is now sleeping. Say 'hey teddy' to wake me up! ---")
                            self.text_queue.put("Good night! Say 'hey teddy' when you want to chat again!")

                        # Process user input if awake
                        elif self.is_awake:
                            print("\nProcessing your request...")
                            openai_thread = threading.Thread(
                                target=send_to_openai_streaming,
                                args=(text, self.text_queue, self.conversation_history, self.history_lock),
                                daemon=True
                            )
                            openai_thread.start()

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


if __name__ == "__main__":
    system = ConversationSystem()
    system.run()
