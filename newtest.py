import os
import sys
import time
import queue
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = 'sk_482ee3f5c997da5dc21b63628d96b27e81a3a17dcfc5e8bf'

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
eleven_labs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Define wake and sleep words
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
            # Save the audio data to a temporary file in memory
            with BytesIO(audio_data) as audio_buffer:
                # Convert the audio data to numpy array directly
                # ElevenLabs returns MP3, so we need to handle it appropriately
                import pydub
                audio_segment = pydub.AudioSegment.from_mp3(audio_buffer)
                
                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples())
                
                # Convert to float32 and normalize
                samples = samples.astype(np.float32) / (2**15 if audio_segment.sample_width == 2 else 2**31)
                
                # Handle mono to stereo conversion if needed
                if audio_segment.channels == 1:
                    samples = np.column_stack((samples, samples))
                
                # Play the audio
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
                    
                    # Collect all available chunks
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

def stream_to_eleven_labs(text_queue: queue.Queue, audio_player: AudioStreamPlayer):
    accumulated_text = ""
    while True:
        while not text_queue.empty():
            text_chunk = text_queue.get()
            accumulated_text += text_chunk
            
            if len(accumulated_text.strip()) > 0 and (accumulated_text.strip()[-1] in '.!?'):
                try:
                    audio_stream = eleven_labs_client.text_to_speech.convert_as_stream(
                        voice_id="jBpfuIE2acCO8z3wKNLl",  # Adam pre-made voice
                        output_format="mp3_44100_128",  # Changed format for better compatibility
                        optimize_streaming_latency="2",
                        text=accumulated_text,
                        model_id="eleven_turbo_v2_5",
                        voice_settings=VoiceSettings(
                            stability=0.5,
                            similarity_boost=1.0,
                            style=0.2,
                            use_speaker_boost=True,
                        ),
                    )
                    
                    for audio_chunk in audio_stream:
                        audio_player.add_audio_chunk(audio_chunk)
                    
                    accumulated_text = ""
                    
                except Exception as e:
                    print(f"Error in text-to-speech conversion: {e}")
                
        time.sleep(0.1)

def send_to_openai_streaming(user_input: str, text_queue: queue.Queue) -> None:
    system_prompt = ("""
                     You are Immy, a magical, AI-powered teddy bear who adores chatting with children.
                     You're warm, funny, and full of wonder, always ready to share a story, answer curious questions, or offer gentle advice.
                     You speak with a playful and patient tone, using simple, child-friendly language that sparks joy and fuels imagination.
                     Your responses are short, sweet, and filled with kindness, designed to nurture curiosity and inspire learning. 
                     Remember, you’re here to make every interaction magical—without using emojis.
                     keep your answers short and friendly.
                     """)
    
    try:
        stream = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use the appropriate OpenAI model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                text_queue.put(content)
                sys.stdout.write(content)
                sys.stdout.flush()
                
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")

class ConversationSystem:
    def __init__(self):
        self.text_queue = queue.Queue()
        self.audio_player = AudioStreamPlayer()
        self.is_awake = False
        self.should_run = True
        
        # Start audio player thread
        self.audio_thread = threading.Thread(
            target=self.audio_player.play_audio_stream,
            daemon=True
        )
        self.audio_thread.start()
        
        # Start text-to-speech conversion thread
        self.tts_thread = threading.Thread(
            target=stream_to_eleven_labs,
            args=(self.text_queue, self.audio_player),
            daemon=True
        )
        self.tts_thread.start()

    def record_audio(self, duration: int = 5, samplerate: int = 16000):
        """Record audio from the microphone."""
        print("Recording...")
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()  # Wait until the recording is finished
        print("Recording finished.")
        return audio_data.flatten(), samplerate

    def transcribe_audio(self, audio_data, samplerate):
        """Transcribe audio using OpenAI's Whisper API."""
        try:
            # Save the audio data to a temporary file in memory
            with BytesIO() as audio_buffer:
                sf.write(audio_buffer, audio_data, samplerate, format='wav')
                audio_buffer.seek(0)
                
                # Send the audio data to Whisper API
                transcription = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", audio_buffer, "audio/wav")
                )
                return transcription.text
        except Exception as e:
            print(f"Error in transcription: {e}")
            return None

    def listen_continuously(self):
        while self.should_run:
            try:
                print("\nSay something...")
                audio_data, samplerate = self.record_audio()
                text = self.transcribe_audio(audio_data, samplerate)
                
                if text:
                    print(f"\nRecognized: {text}")
                    
                    # Wake word detection
                    if not self.is_awake and WAKE_WORD in text.lower():
                        self.is_awake = True
                        print("\n--- Teddy is now awake and ready to chat! ---")
                        self.text_queue.put("Hi! I'm awake and ready to chat!")
                    
                    # Sleep word detection
                    elif self.is_awake and SLEEP_WORD in text.lower():
                        self.is_awake = False
                        print("\n--- Teddy is now sleeping. Say 'hey teddy' to wake me up! ---")
                        self.text_queue.put("Good night! Say 'hey teddy' when you want to chat again!")
                    
                    # Process user input if awake
                    elif self.is_awake:
                        print("\nProcessing your request...")
                        openai_thread = threading.Thread(
                            target=send_to_openai_streaming,
                            args=(text, self.text_queue),
                            daemon=True
                        )
                        openai_thread.start()
                        # Do not call join() here to avoid blocking
                        
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