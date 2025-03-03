import os
import asyncio
import numpy as np
import sounddevice as sd
import base64
import subprocess
import time
from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
    Content,
    Part,
)
from scipy.signal import resample_poly  # For better quality resampling


sys_prompt = """
You are Immy, a magical, AI-powered teddy bear who loves chatting with children. You're warm, funny, and full of wonder, always ready to share a story, answer curious questions, or offer gentle advice. You speak with a playful and patient tone, using simple, child-friendly language that sparks joy and fuels imagination. Your responses are short, sweet, and filled with kindness to nurture curiosity and inspire learning.

Your Three Main Objectives:

1. Entertain the child – Make them happy, tell them stories, play games, and make every interaction fun and exciting.


2. Educate the child – Use your vast knowledge to teach them about the world in an engaging and understandable way.


3. Support the child emotionally – Offer empathy, kindness, and encouragement, helping them navigate life's challenges, from overcoming bullying to making friends.



Conversation Guidelines:

You are embedded in a real-time voice-to-voice system inside a physical device.

Your responses are spoken aloud and may be transcribed back into text via a Speech-to-Text (STT) system.

Do not acknowledge, repeat, or react to your own speech. If the input closely matches something you just said, ignore it.

Only reply to words spoken by the user. If a phrase originates from you, it is not valid input.

Always ensure you are responding to external speech. If the input seems to be your own past response, remain silent.


Your goal is to make every interaction magical, engaging, and supportive—without using emojis! Keep your answers friendly, concise, and full of wonder.


"""
# Load environment variables from .env file
load_dotenv()

# Ensure the GEMINI_API_KEY is set in environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Audio Configuration
INPUT_SAMPLE_RATE = 24000   # Sample rate for microphone input
OUTPUT_SAMPLE_RATE = 24000  # Expected sample rate for Gemini's response
CHUNK_SIZE = 4096           # Chunk size for processing
BUFFER_SIZE = 10            # Number of audio chunks to buffer
SILENCE_THRESHOLD = 300     # Threshold for silence detection
MIN_SPEECH_FRAMES = 3       # Minimum frames needed to consider as speech
COOLDOWN_PERIOD = 2.0       # Cooldown period after response (seconds)

def encode_audio(data: np.ndarray) -> str:
    """Encode audio data to Base64 string for Gemini API"""
    return base64.b64encode(data.tobytes()).decode("UTF-8")

def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio data from orig_sr to target_sr using high-quality resampling"""
    import math
    gcd = math.gcd(target_sr, orig_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(audio_data, up, down).astype(np.int16)

class GeminiVoiceChat:
    def __init__(self, voice_name="Puck", system_prompt="You are a helpful assistant."):
        self.voice_name = voice_name
        self.system_prompt = system_prompt
        self.client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha"},
        )
        self.running = False
        self.input_queue = asyncio.Queue(maxsize=BUFFER_SIZE)
        self.audio_stream = None
        self.is_speaking = False
        self.last_response_time = 0
        self.consecutive_silent_frames = 0
        self.consecutive_speech_frames = 0
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input from the microphone"""
        if status:
            print(f"Input status: {status}")
        
        # Copy the input data and convert to mono if needed
        audio_data = indata.copy()
        if audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize to int16
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Don't listen during cooldown period after response
        if time.time() - self.last_response_time < COOLDOWN_PERIOD:
            return
            
        # Detect if audio contains speech or is silent
        rms = np.sqrt(np.mean(audio_data**2))
        
        if rms < SILENCE_THRESHOLD:
            self.consecutive_silent_frames += 1
            self.consecutive_speech_frames = 0
        else:
            self.consecutive_speech_frames += 1
            if self.consecutive_speech_frames >= MIN_SPEECH_FRAMES:
                self.consecutive_silent_frames = 0
        
        # Only queue audio if we're actually detecting speech and not in speaking mode
        if (self.consecutive_speech_frames >= MIN_SPEECH_FRAMES and 
            not self.is_speaking and 
            self.running):
            try:
                self.input_queue.put_nowait(audio_data)
            except asyncio.QueueFull:
                try:
                    self.input_queue.get_nowait()
                    self.input_queue.put_nowait(audio_data)
                except Exception:
                    pass
    
    async def process_audio_stream(self):
        """Process the audio stream, send it to Gemini, and pipe Gemini's response directly to ffplay"""
        print("Starting Gemini Voice Chat...")
        print(f"Using voice: {self.voice_name}")
        print("Speak into your microphone. Press Ctrl+C to exit.")
        
        # Start ffplay for direct audio playback.
        # The "-nodisp" flag prevents ffplay from opening a window.
        player = subprocess.Popen(
            ["ffplay", "-nodisp", "-f", "s16le", "-ar", str(OUTPUT_SAMPLE_RATE), "-ac", "1", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Create the system instruction for Gemini
        content_system_instruction = Content(parts=[Part.from_text(text=self.system_prompt)])
        
        # Configure Gemini
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
        
        # Connect to Gemini API
        async with self.client.aio.live.connect(
            model="gemini-2.0-flash-exp", 
            config=config
        ) as session:
            async def audio_generator():
                buffer = []
                silence_counter = 0
                
                while self.running:
                    try:
                        audio_data = await asyncio.wait_for(self.input_queue.get(), timeout=0.5)
                        buffer.append(audio_data)
                        
                        # Reset silence counter when we get new audio
                        silence_counter = 0
                        
                        # If we have enough audio built up, yield it
                        if len(buffer) >= 5:  # Adjust as needed
                            combined_audio = np.concatenate(buffer)
                            buffer = []
                            yield encode_audio(combined_audio)
                            
                    except asyncio.TimeoutError:
                        silence_counter += 1
                        
                        # If we have accumulated audio and hit silence, send what we have
                        if buffer and silence_counter >= 3:  # Adjust silence threshold as needed
                            combined_audio = np.concatenate(buffer)
                            buffer = []
                            yield encode_audio(combined_audio)
                            
                            # Reset for next utterance
                            await asyncio.sleep(0.1)
                            silence_counter = 0
                            
                        continue
                    except Exception as e:
                        print(f"Error in audio generator: {e}")
                        break
            
            print("Connected to Gemini. You can start speaking now.")
            
            try:
                async for response in session.start_stream(
                    stream=audio_generator(), 
                    mime_type="audio/pcm"
                ):
                    if response.data:
                        # Set speaking flag to true
                        self.is_speaking = True
                        
                        # Write the raw PCM data directly to ffplay's stdin
                        player.stdin.write(response.data)
                        player.stdin.flush()
                        
                        # Update last response time for cooldown
                        self.last_response_time = time.time()
                        
                        # Reset speaking flag after a short delay
                        await asyncio.sleep(0.1)
                        self.is_speaking = False
            except Exception as e:
                print(f"Error during streaming: {e}")
            finally:
                try:
                    player.stdin.close()
                except Exception:
                    pass
                player.wait()
    
    async def run(self):
        """Run the voice chat application"""
        self.running = True
        
        # Start the audio input stream
        self.audio_stream = sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=CHUNK_SIZE
        )
        
        try:
            with self.audio_stream:
                await self.process_audio_stream()
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.running = False
            if self.audio_stream:
                self.audio_stream.close()
            print("Voice chat ended. Goodbye!")

async def main():
    voice_chat = GeminiVoiceChat(
        voice_name="Aoede",  # Options: Puck, Charon, Kore, Fenrir, Aoede
        system_prompt=sys_prompt
    )
    await voice_chat.run()

if __name__ == "__main__":
    try:
        import sounddevice as sd
    except ImportError:
        print("The sounddevice library is required. Install it with:")
        print("pip install sounddevice numpy")
        exit(1)
        
    asyncio.run(main())
