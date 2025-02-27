import os
import sys
import time
import queue
import threading
import sounddevice as sd
import speech_recognition as sr
from io import BytesIO
from typing import Optional, AsyncGenerator
import numpy as np
import asyncio
import tempfile
import base64

# For audio file handling
import soundfile as sf
from pydub import AudioSegment

# For environment variables (used for Gemini API key)
from dotenv import load_dotenv

load_dotenv()

# Import Gemini AI
import google.generativeai as genai
from google.generativeai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
    Content,
    Part,
)

# Retrieve the Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Define wake and sleep words
WAKE_WORD = "hey teddy"  # Fixed wake word to match the print statement
SLEEP_WORD = "good night"


def encode_audio(data: np.ndarray) -> str:
    """Encode Audio data to send to Gemini"""
    return base64.b64encode(data.tobytes()).decode("UTF-8")


class AudioStreamPlayer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_finished_callback: Optional[callable] = None  # Callback after playback finishes

    def add_audio_chunk(self, chunk):
        if chunk is not None:
            self.audio_queue.put(chunk)

    def play_audio_array(self, sample_rate, samples):
        try:
            # Play the audio from numpy array
            self.is_playing = True
            sd.play(samples, sample_rate)
            sd.wait()
            self.is_playing = False

            # Call the playback finished callback if set
            if self.playback_finished_callback:
                self.playback_finished_callback()

        except Exception as e:
            print(f"Error playing audio: {e}")
            self.is_playing = False

    def play_audio_stream(self):
        while True:
            if not self.audio_queue.empty():
                try:
                    audio_data = self.audio_queue.get()
                    if isinstance(audio_data, tuple):
                        # Tuple of (sample_rate, samples)
                        sample_rate, samples = audio_data
                        self.play_audio_array(sample_rate, samples)
                except Exception as e:
                    print(f"Error in audio playback: {e}")
                    self.is_playing = False
            time.sleep(0.1)


class GeminiHandler:
    """Handler for the Gemini API"""

    def __init__(self, voice_name="Puck", system_prompt=None):
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.quit = asyncio.Event()
        self.voice_name = voice_name
        self.system_prompt = system_prompt or (
            "You are Immy, a magical, AI-powered teddy bear who adores chatting with children. "
            "You're warm, funny, and full of wonder, always ready to share a story, answer curious questions, or offer gentle advice. "
            "You speak with a playful and patient tone, using simple, child-friendly language that sparks joy and fuels imagination. "
            "Your responses are short, sweet, and filled with kindness, designed to nurture curiosity and inspire learning. "
            "Remember, you're here to make every interaction magical—without using emojis. "
            "Keep your answers short and friendly."
        )
        self.output_sample_rate = 24000  # Gemini's default output sample rate

    async def start_session(self):
        client = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

        # Wrap the system prompt in a Content object
        content_system_instruction = Content(parts=[Part.from_text(text=self.system_prompt)])

        config = LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=self.voice_name,
                    )
                )
            ),
            system_instruction=content_system_instruction
        )

        try:
            async with client.start_live_connect(config=config) as session:
                async for audio in session.send_audio_stream(
                        stream=self.stream(), mime_type="audio/pcm"
                ):
                    if audio.data:
                        array = np.frombuffer(audio.data, dtype=np.int16)
                        await self.output_queue.put((self.output_sample_rate, array))
        except Exception as e:
            print(f"Error in Gemini session: {e}")
            self.quit.set()

    async def stream(self) -> AsyncGenerator[bytes, None]:
        while not self.quit.is_set():
            try:
                audio = await asyncio.wait_for(self.input_queue.get(), 0.1)
                yield audio
            except (asyncio.TimeoutError, TimeoutError):
                pass
            except Exception as e:
                print(f"Error in audio stream: {e}")
                # Continue the stream despite errors
                pass

    async def add_audio(self, audio_array):
        """Add audio data to the input queue"""
        try:
            audio_message = encode_audio(audio_array)
            await self.input_queue.put(audio_message)
        except Exception as e:
            print(f"Error encoding or adding audio: {e}")

    async def get_audio(self):
        """Get audio data from the output queue"""
        return await self.output_queue.get()


# ------------------------------
# Conversation System Class
# ------------------------------
class ConversationSystem:
    def __init__(self):
        self.audio_player = AudioStreamPlayer()
        self.is_awake = False
        self.should_run = True
        self.recognizer = sr.Recognizer()

        # Timestamp for when the last TTS utterance finished playing
        self.last_tts_end = 0.0
        # Minimum delay (in seconds) after TTS playback before processing microphone input
        self.post_tts_cooldown = 1.0

        # Set the callback so that when audio playback finishes, we update last_tts_end
        self.audio_player.playback_finished_callback = self.audio_finished_callback

        # Create Gemini handler
        self.gemini_handler = GeminiHandler()

        # Start audio player thread
        self.audio_thread = threading.Thread(
            target=self.audio_player.play_audio_stream,
            daemon=True
        )
        self.audio_thread.start()

        # Start the Gemini session
        self.gemini_task = None

    def audio_finished_callback(self):
        self.last_tts_end = time.time()

    async def process_gemini_output(self):
        """Process output audio from Gemini and send to audio player"""
        while self.should_run:
            try:
                if self.gemini_handler.quit.is_set():
                    print("Gemini session has ended")
                    break
                    
                audio_data = await self.gemini_handler.get_audio()
                self.audio_player.add_audio_chunk(audio_data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in Gemini output processing: {e}")
                await asyncio.sleep(0.1)

    async def start_gemini_session(self):
        """Start the Gemini session and output processor"""
        try:
            self.gemini_task = asyncio.create_task(self.gemini_handler.start_session())
            output_task = asyncio.create_task(self.process_gemini_output())
            return output_task
        except Exception as e:
            print(f"Error starting Gemini session: {e}")
            return None

    async def send_audio_to_gemini(self, audio_data):
        """Send audio data to Gemini"""
        if not self.gemini_handler.quit.is_set():
            await self.gemini_handler.add_audio(audio_data)

    def listen_continuously(self):
        """Main loop for listening to user input"""
        # Start the asyncio event loop for Gemini
        asyncio.run(self._listen_continuously())

    async def _listen_continuously(self):
        """Async implementation of the listening loop"""
        # Start Gemini session
        output_task = await self.start_gemini_session()
        
        if not output_task:
            print("Failed to start Gemini session, exiting")
            return

        with sr.Microphone() as source:
            print("\nListening...")
            self.recognizer.adjust_for_ambient_noise(source)

            while self.should_run:
                if self.audio_player.is_playing:
                    await asyncio.sleep(0.1)
                    continue

                if time.time() - self.last_tts_end < self.post_tts_cooldown:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    print("\nSay something...")
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)

                    # Convert audio to numpy array for Gemini
                    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

                    try:
                        # For wake/sleep word detection, still use Google Speech Recognition
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"\nRecognized: {text}")

                        # Wake word detection
                        if not self.is_awake and WAKE_WORD in text:
                            self.is_awake = True
                            print("\n--- Teddy is now awake and ready to chat! ---")
                            # Send the wake word audio to trigger response
                            await self.send_audio_to_gemini(audio_data)

                        # Sleep word detection
                        elif self.is_awake and SLEEP_WORD in text:
                            self.is_awake = False
                            print("\n--- Teddy is now sleeping. Say 'hey teddy' to wake me up! ---")
                            # Send sleep command audio to trigger response
                            await self.send_audio_to_gemini(audio_data)

                        # Process user input if awake
                        elif self.is_awake:
                            print("\nProcessing your request...")
                            # Send the audio directly to Gemini
                            await self.send_audio_to_gemini(audio_data)
                        else:
                            print("\nTeddy is sleeping. Say 'hey teddy' to wake me up!")

                    except sr.UnknownValueError:
                        print("\nCould not understand audio.")
                        continue
                    except sr.RequestError as e:
                        print(f"\nCould not request results from Google Speech Recognition service: {e}")
                        continue

                except KeyboardInterrupt:
                    self.should_run = False
                    break
                except Exception as e:
                    print(f"Error in audio processing: {e}")

        # Clean up
        print("\nCleaning up...")
        self.gemini_handler.quit.set()
        if self.gemini_task:
            self.gemini_task.cancel()
        if output_task:
            output_task.cancel()

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
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.should_run = False


if __name__ == "__main__":
    try:
        system = ConversationSystem()
        system.run()
    except Exception as e:
        print(f"Error running conversation system: {e}")
        sys.exit(1)