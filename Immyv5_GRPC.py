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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
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

def send_to_openai_streaming(user_input: str, text_queue: queue.Queue):
    messages = [{"role": "system", "content": "You are a friendly AI teddy bear."},
                {"role": "user", "content": user_input}]
    try:
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                text_queue.put(chunk.choices[0].delta.content)
    except Exception as e:
        print(f"Error in OpenAI call: {e}")

class ConversationSystem:
    def __init__(self):
        self.text_queue = queue.Queue()
        self.audio_player = AudioStreamPlayer()
        self.is_awake = False
        self.should_run = True
        self.recognizer = sr.Recognizer()
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
                        self.text_queue.put("Good night!")
                    elif self.is_awake:
                        openai_thread = threading.Thread(target=send_to_openai_streaming, args=(text, self.text_queue), daemon=True)
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
