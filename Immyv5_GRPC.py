import os
import sys
import time
import queue
import threading
import asyncio
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from io import BytesIO
from typing import Optional
import grpc
import tempfile
import soundfile as sf
from pydub import AudioSegment
from dotenv import load_dotenv
import edge_tts
from openai import OpenAI

# Load environment variables and initialize OpenAI.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Define wake and sleep words.
WAKE_WORD = "hey "
SLEEP_WORD = "good night"

class AudioStreamPlayer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False

    def add_audio_chunk(self, chunk):
        if chunk:
            self.audio_queue.put(chunk)

    def play_audio_file(self, audio_data):
        try:
            with BytesIO(audio_data) as audio_buffer:
                audio_segment = AudioSegment.from_mp3(audio_buffer)
                samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / (2**15)
                if audio_segment.channels == 1:
                    samples = np.column_stack((samples, samples))
                self.is_playing = True
                sd.play(samples, audio_segment.frame_rate)
                sd.wait()
                self.is_playing = False
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.is_playing = False

    def play_audio_stream(self):
        while True:
            if not self.is_playing and not self.audio_queue.empty():
                try:
                    audio_data = self.audio_queue.get()
                    self.play_audio_file(audio_data)
                except Exception as e:
                    print(f"Error in audio playback: {e}")
            time.sleep(0.1)

async def async_text_to_speech(text: str, voice="en-US-AnaNeural", rate=25, pitch=10) -> bytes:
    if not text.strip():
        return None
    rate_str, pitch_str = f"{rate:+d}%", f"{pitch:+d}Hz"
    communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
    await communicate.save(tmp_path)
    with open(tmp_path, "rb") as f:
        audio_data = f.read()
    os.remove(tmp_path)
    return audio_data

def text_to_speech_sync(text: str) -> bytes:
    return asyncio.run(async_text_to_speech(text))

def stream_to_edge_tts(text_queue: queue.Queue, audio_player: AudioStreamPlayer):
    while True:
        if audio_player.is_playing:
            time.sleep(0.1)
            continue
        while not text_queue.empty():
            text_chunk = text_queue.get()
            if text_chunk.strip():
                try:
                    audio_data = text_to_speech_sync(text_chunk)
                    if audio_data:
                        audio_player.add_audio_chunk(audio_data)
                except Exception as e:
                    print(f"Error in text-to-speech: {e}")
        time.sleep(0.1)

def send_to_openai_streaming(user_input: str, text_queue: queue.Queue, conversation_history: list, history_lock: threading.Lock) -> None:
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
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        answer_text = ""
        accumulated_text = ""
        CHUNK_WORD_THRESHOLD = 10  # Flush when this many words have accumulated.
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                answer_text += content
                accumulated_text += content
                # Check word count in the accumulated text.
                if len(accumulated_text.strip().split()) >= CHUNK_WORD_THRESHOLD:
                    text_queue.put(accumulated_text)
                    sys.stdout.write(accumulated_text)
                    sys.stdout.flush()
                    accumulated_text = ""
        # Flush any remaining text.
        if accumulated_text:
            text_queue.put(accumulated_text)
            sys.stdout.write(accumulated_text)
            sys.stdout.flush()
        with history_lock:
            conversation_history.append({"role": "assistant", "content": answer_text})
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")

class ConversationSystem:
    def __init__(self):
        self.text_queue = queue.Queue()
        self.audio_player = AudioStreamPlayer()
        self.is_awake = False
        self.should_run = True
        self.recognizer = sr.Recognizer()
        self.conversation_history = []
        self.history_lock = threading.Lock()
        self.audio_thread = threading.Thread(target=self.audio_player.play_audio_stream, daemon=True)
        self.audio_thread.start()
        self.tts_thread = threading.Thread(target=stream_to_edge_tts, args=(self.text_queue, self.audio_player), daemon=True)
        self.tts_thread.start()

    def listen_continuously(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.should_run:
                if self.audio_player.is_playing:
                    time.sleep(0.1)
                    continue
                try:
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
                    text = self.recognizer.recognize_google(audio).lower()
                    if not self.is_awake and WAKE_WORD in text:
                        self.is_awake = True
                        self.text_queue.put("Hi! I'm awake and ready to chat!")
                    elif self.is_awake and SLEEP_WORD in text:
                        self.is_awake = False
                        self.text_queue.put("Good night! Say 'hey teddy' when you want to chat again!")
                    elif self.is_awake:
                        openai_thread = threading.Thread(
                            target=send_to_openai_streaming,
                            args=(text, self.text_queue, self.conversation_history, self.history_lock),
                            daemon=True
                        )
                        openai_thread.start()
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")

    def run(self):
        try:
            self.listen_continuously()
        except KeyboardInterrupt:
            self.should_run = False

if __name__ == "__main__":
    system = ConversationSystem()
    system.run()