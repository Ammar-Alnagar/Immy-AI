#!/usr/bin/env python3
import asyncio
import os
import sys
import struct
import time
import traceback
import pyaudio
from google import genai
from google.genai.types import LiveConnectConfig, HttpOptions, Modality
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
    CONFIG = {"generation_config": {"response_modalities": ["AUDIO"]}}

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

class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None  # Gemini's audio for playback
        self.out_queue = None       # Processed mic audio for sending to Gemini
        self.session = None         # Gemini API session
        self.mic_stream = None      # Microphone stream
        self.reverse_stream = None  # Reverse (playback reference) stream

        # Variables to prevent self-listening
        self.last_playback_end = 0.0
        self.playback_cooldown = 3.0  # Seconds to wait after playback before processing mic

        # Initialize WebRTC Audio Processing (AEC/NS/VAD)
        self.ap = None
        if AP:
            try:
                # Enable NS, VAD and AEC if supported (remove enable_aec if not supported)
                self.ap = AP(enable_ns=True, enable_vad=True, enable_aec=True)
                self.ap.set_stream_format(SEND_SAMPLE_RATE, CHANNELS)
                self.ap.set_ns_level(3)
                self.ap.set_vad_level(3)
                # Optionally: self.ap.set_aec_level(3)
            except Exception as e:
                print(f"Error initializing audio processing: {e}")
        else:
            print("No WebRTC AudioProcessing module available; echo cancellation disabled.")

    async def listen_mic_audio(self):
        """Capture microphone audio, process with AEC (if available), then queue if voice is detected.
           Suppress mic input if playback occurred recently."""
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
            # Check for cooldown – if playback ended recently, skip processing mic input
            if time.time() - self.last_playback_end < self.playback_cooldown:
                await asyncio.sleep(0.1)
                continue

            try:
                data = await asyncio.to_thread(self.mic_stream.read, FRAME_SIZE_BYTES_16K, exception_on_overflow=False)
            except Exception as e:
                print(f"Error reading from microphone: {e}")
                continue

            if self.ap:
                try:
                    processed_mic_data = self.ap.process_stream(data)
                    if not self.ap.has_voice():
                        continue
                    await self.out_queue.put({"data": processed_mic_data, "mime_type": "audio/pcm"})
                except Exception as e:
                    print(f"AEC error (mic): {e}")
                    await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            else:
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def listen_reverse_audio(self):
        """
        Capture playback (reverse) audio from the loopback device.
        Open the device using its native channel count and convert to mono if needed.
        """
        try:
            rev_info = pya.get_device_info_by_index(reverse_device_index)
            print(f"Reverse device info: {rev_info}")
            native_channels = int(rev_info.get("maxInputChannels", 0))
            if native_channels <= 0:
                print(f"Device {reverse_device_index} is not a valid input device.")
                return
            frames_per_buffer = FRAME_SAMPLES_16K * native_channels * 2  # 2 bytes per sample
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

    async def receive_audio(self):
        """Receive Gemini's audio responses."""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if response.data:
                    print(f"Received {len(response.data)} bytes from Gemini")
                    self.audio_in_queue.put_nowait(response.data)
                if response.text:
                    print("Gemini:", response.text, end="")
            print()

    async def play_audio(self):
        """Play Gemini's audio responses on your output device and update cooldown."""
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
                # Update the playback cooldown timestamp after playing a chunk
                self.last_playback_end = time.time()
            except Exception as e:
                print(f"Error writing to output device: {e}")

    async def send_realtime(self):
        """Send processed microphone audio to Gemini."""
        while True:
            msg = await self.out_queue.get()
            try:
                await self.session.send(input=msg)
            except Exception as e:
                print(f"Error sending realtime data: {e}")

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                
                print("Voice chat started. Speak into your microphone. Press Ctrl+C to quit.")
                print("Note: For reliable echo removal, use headphones or a proper loopback device.")

                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_mic_audio())
                tg.create_task(self.listen_reverse_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await asyncio.Future()  # Run indefinitely
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"An error occurred in run: {e}")
        finally:
            print("Voice chat session ended.")

if __name__ == "__main__":
    try:
        loop = AudioLoop()
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        print("\nChat terminated by user.")
    finally:
        pya.terminate()
        print("Audio resources released.")