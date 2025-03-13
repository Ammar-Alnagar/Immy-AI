#!/usr/bin/env python3
import asyncio
import os
import sys
import struct
import time
import traceback
import pyaudio
import pyttsx3  # Text-to-speech engine
from google import genai
from google.genai.types import LiveConnectConfig, HttpOptions, Modality
# For voice configuration:
from google.genai.types import SpeechConfig, VoiceConfig, PrebuiltVoiceConfig, Content, Part
from dotenv import load_dotenv

# Suppress ALSA warnings (set log level to 0)
os.environ["ALSA_LOGLEVEL"] = "0"

try:
    from webrtc_audio_processing import AudioProcessingModule as AP
except ImportError:
    AP = None
    print("WebRTC Audio Processing module not found. Build/install from GitHub.")

load_dotenv()

if sys.version_info < (3, 11, 0):
    print("Error: This script requires Python 3.11 or newer.")
    sys.exit(1)

# Configuration constants
FORMAT = pyaudio.paInt16
CHANNELS = 1  # processing in mono

# Mic and reverse processing sample rate remains 16000 Hz for AEC.
SEND_SAMPLE_RATE = 16000
# Set playback (output) sample rate (adjust as needed)
RECEIVE_SAMPLE_RATE = 24000

# Use 20ms frames
FRAME_DURATION_MS = 20
FRAME_SAMPLES_16K = int(SEND_SAMPLE_RATE * FRAME_DURATION_MS / 1000)
FRAME_SIZE_BYTES_16K = FRAME_SAMPLES_16K * 2  # for mono 16-bit

# For output, recalc frame size based on playback rate.
FRAME_SAMPLES_OUTPUT = int(RECEIVE_SAMPLE_RATE * FRAME_DURATION_MS / 1000)
FRAME_SIZE_BYTES_OUTPUT = FRAME_SAMPLES_OUTPUT * 2

# Device configuration (update these to match your system)
mic_device_index = 0         # Your physical microphone (near-end)
reverse_device_index = 3     # Loopback/virtual input capturing playback (far-end reference)
output_device_index = 1      # Your playback device (speakers/headphones)

use_vertexai = False
PROJECT_ID = 'set-me-up'

if use_vertexai:
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location='us-central1',
        http_options=HttpOptions(api_version="v1beta1")
    )
    MODEL = "gemini-2.0-flash-exp"
    CONFIG = LiveConnectConfig(response_modalities=[Modality.AUDIO])
else:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is not set in .env")
        sys.exit(1)
    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1alpha"}
    )
    MODEL = "models/gemini-2.0-flash-exp"
    # We rebuild CONFIG as a LiveConnectConfig in the GeminiVoiceChat constructor.
    CONFIG = None

# Create a global PyAudio instance
pya = pyaudio.PyAudio()

def stereo_to_mono(data):
    """
    Convert stereo 16-bit little-endian data to mono by extracting the left channel.
    Assumes data length is a multiple of 4 bytes.
    """
    num_samples = len(data) // 2
    fmt = "<" + "h" * num_samples
    samples = struct.unpack(fmt, data)
    mono_samples = samples[0:num_samples:2]
    fmt_mono = "<" + "h" * len(mono_samples)
    return struct.pack(fmt_mono, *mono_samples)

# Updated function to speak announcement using a selected voice
def speak_announcement(text):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        # Print available voices for reference.
        print("Available voices:")
        for i, voice in enumerate(voices):
            print(f"Voice {i}: {voice.name} - {voice.id}")
        # Change the index below to select a different voice.
        # For example, if index 1 sounds better, use that; otherwise try 0 or 2.
        desired_index = 23 if len(voices) > 23 else 0
        engine.setProperty("voice", voices[desired_index].id)
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()  # Blocks until speech is done
        engine.stop()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None  # For Gemini's audio responses (for local playback)
        self.out_queue = None       # Processed mic audio for sending to Gemini
        self.session = None         # Gemini API session
        self.mic_stream = None      # Microphone stream
        self.reverse_stream = None  # Reverse (playback reference) stream

        # Variables to prevent self-listening
        self.last_playback_end = 0.0
        self.playback_cooldown = 0.3  # Seconds to wait after playback before processing mic

        # Initialize WebRTC Audio Processing (AEC/NS/VAD)
        self.ap = None
        if AP:
            try:
                self.ap = AP(enable_ns=True, enable_vad=True)
                self.ap.set_stream_format(SEND_SAMPLE_RATE, CHANNELS)
                self.ap.set_ns_level(3)
                self.ap.set_vad_level(3)
            except Exception as e:
                print(f"Error initializing audio processing: {e}")
        else:
            print("No WebRTC AudioProcessing module available; echo cancellation disabled.")

    async def listen_mic_audio(self):
        async def open_mic_stream():
            try:
                return await asyncio.to_thread(
                    pya.open,
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SEND_SAMPLE_RATE,
                    input=True,
                    input_device_index=mic_device_index,
                    frames_per_buffer=FRAME_SIZE_BYTES_16K,
                )
            except Exception as e:
                print(f"Error opening microphone (device {mic_device_index}): {e}")
                return None

        self.mic_stream = await open_mic_stream()
        if not self.mic_stream:
            return

        while True:
            if time.time() - self.last_playback_end < self.playback_cooldown:
                await asyncio.sleep(0.1)
                continue

            try:
                data = await asyncio.to_thread(
                    self.mic_stream.read, FRAME_SIZE_BYTES_16K, exception_on_overflow=False
                )
            except Exception as e:
                print(f"Error reading from microphone: {e}")
                continue

            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def listen_reverse_audio(self):
        try:
            rev_info = pya.get_device_info_by_index(reverse_device_index)
            print(f"Reverse device info: {rev_info}")
            native_channels = int(rev_info.get("maxInputChannels", 0))
            if native_channels <= 0:
                print(f"Device {reverse_device_index} is not a valid input device.")
                return
            frames_per_buffer = FRAME_SAMPLES_16K * native_channels * 2
            self.reverse_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=native_channels,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=reverse_device_index,
                frames_per_buffer=frames_per_buffer,
            )
        except Exception as e:
            print(f"Error opening reverse audio device (device {reverse_device_index}): {e}")
            return

        while True:
            try:
                rev_data = await asyncio.to_thread(self.reverse_stream.read, frames_per_buffer)
            except Exception as e:
                print(f"Error reading from reverse audio device: {e}")
                continue

            if native_channels != 1:
                try:
                    rev_data = stereo_to_mono(rev_data)
                except Exception as e:
                    print(f"Error converting reverse audio to mono: {e}")
                    continue

            if self.ap:
                try:
                    self.ap.process_reverse_stream(rev_data)
                except Exception as e:
                    print(f"Reverse-stream processing error: {e}")

    async def play_audio(self):
        try:
            out_info = pya.get_device_info_by_index(output_device_index)
            print(f"Output device info: {out_info}")
            stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
                output_device_index=output_device_index,
                frames_per_buffer=FRAME_SIZE_BYTES_OUTPUT,
            )
        except Exception as e:
            print(f"Error opening output device (device {output_device_index}): {e}")
            return

        while True:
            data = await self.audio_in_queue.get()
            if not data:
                continue
            try:
                await asyncio.to_thread(stream.write, data)
                print(f"Played {len(data)} bytes")
                self.last_playback_end = time.time()
            except Exception as e:
                print(f"Error writing to output device: {e}")

    async def run(self):
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                self.out_queue = asyncio.Queue(maxsize=5)
                self.audio_in_queue = asyncio.Queue()

                print("Voice chat started. Speak into your microphone. Press Ctrl+C to quit.")
                print("Note: For reliable echo removal, use headphones or a proper loopback device.")

                mic_task = asyncio.create_task(self.listen_mic_audio())
                reverse_task = asyncio.create_task(self.listen_reverse_audio())
                playback_task = asyncio.create_task(self.play_audio())

                async def audio_generator():
                    while True:
                        try:
                            msg = await self.out_queue.get()
                            yield msg["data"]
                        except Exception as e:
                            print(f"Error in audio generator: {e}")
                            break

                async for response in session.start_stream(
                    stream=audio_generator(),
                    mime_type="audio/pcm"
                ):
                    if response.data:
                        print(f"Received {len(response.data)} bytes from Gemini")
                        await self.audio_in_queue.put(response.data)
                    if response.text:
                        print("Gemini:", response.text, end="")

                mic_task.cancel()
                reverse_task.cancel()
                playback_task.cancel()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"An error occurred in run: {e}")
        finally:
            print("Voice chat session ended.")

# --- Define GeminiVoiceChat with voice configuration ---
sys_prompt = """
You are Immy, a magical, AI-powered teddy bear who loves chatting with children. You’re warm, funny, and full of wonder, always ready to share a story, answer curious questions, or offer gentle advice.
"""

class GeminiVoiceChat(AudioLoop):
    def __init__(self, voice_name: str, system_prompt: str):
        global CONFIG
        self.voice_name = voice_name
        self.system_prompt = system_prompt
        super().__init__()
        # Always rebuild CONFIG as a LiveConnectConfig object.
        CONFIG = LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=self.voice_name,
                    )
                )
            ),
            system_instruction=Content(parts=[Part.from_text(text=self.system_prompt)])
        )

def main():
    announcement_text = "chat is ready"
    print("Announcement:", announcement_text)
    speak_announcement(announcement_text)
    print("Announcement complete. Starting Gemini session.")
    
    # Allow a brief pause for audio resources to settle
    time.sleep(5)
    
    # Reinitialize PyAudio to ensure it is free for the Gemini session.
    global pya
    pya.terminate()
    pya = pyaudio.PyAudio()
    
    try:
        voice_chat = GeminiVoiceChat(
            voice_name="Aoede",  # Options: Puck, Charon, Kore, Fenrir, Aoede
            system_prompt=sys_prompt
        )
        asyncio.run(voice_chat.run())
    except KeyboardInterrupt:
        print("\nChat terminated by user.")
    finally:
        pya.terminate()
        print("Audio resources released.")

if __name__ == "__main__":
    main()
