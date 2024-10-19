import os
import time
import tkinter as tk
from tkinter import messagebox, ttk
import json
import sounddevice as sd
import numpy as np
import wave
from subprocess import Popen, PIPE
import pyttsx3
from llama_cpp import Llama

class SpeechBot:
    def __init__(self, 
                 llama_model_path="models/llama-2-7b-q4_0.gguf",  # Quantized model
                 whisper_model_path="models/whisper-tiny-q5_1.bin"):  # Tiny quantized model
        
        # Initialize llama.cpp with minimal resources
        self.llm = Llama(
            model_path=llama_model_path,
            n_ctx=512,  # Reduced context window
            n_threads=2,  # Reduced threads for Raspberry Pi
            n_batch=8    # Smaller batch size
        )

        # Path to whisper.cpp executable
        self.whisper_path = "./whisper"
        self.whisper_model = whisper_model_path

        # Initialize text-to-speech
        try:
            self.engine = pyttsx3.init()
            # Lower speech rate for better performance
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            self.voices = self.engine.getProperty('voices')
            self.current_voice_idx = 0
            if self.voices:
                self.engine.setProperty('voice', self.voices[0].id)
        except Exception as e:
            print(f"TTS initialization error: {e}")
            # Fallback to espeak if pyttsx3 fails
            self.use_espeak = True
        else:
            self.use_espeak = False

        # Audio recording settings optimized for Raspberry Pi
        self.sample_rate = 16000
        self.recording = False
        self.audio_data = []
        
        # Create necessary directories
        os.makedirs("temp", exist_ok=True)

    def set_voice_properties(self, rate, volume, voice_idx):
        """Update voice properties"""
        if not self.use_espeak:
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            if 0 <= voice_idx < len(self.voices):
                self.engine.setProperty('voice', self.voices[voice_idx].id)
                self.current_voice_idx = voice_idx

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if self.recording:
            self.audio_data.append(indata.copy())

    def start_recording(self):
        """Start audio recording"""
        self.recording = True
        self.audio_data = []
        self.record_button.config(text="Stop Talking", bg="#FF4444")
        
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=1024,  # Smaller blocksize for better performance
                dtype=np.float32
            )
            self.stream.start()
        except Exception as e:
            print(f"Recording error: {e}")
            messagebox.showerror("Error", "Could not start recording")
            self.recording = False
            self.record_button.config(text="Start Talking", bg="#4CAF50")

    def stop_recording(self):
        """Stop recording and process the audio"""
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            self.record_button.config(text="Start Talking", bg="#4CAF50")

            if len(self.audio_data) > 0:
                # Process audio in temp directory
                temp_wav = "temp/recording.wav"
                try:
                    audio_data = np.concatenate(self.audio_data)
                    with wave.open(temp_wav, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

                    # Use whisper.cpp command line for transcription
                    cmd = [
                        self.whisper_path,
                        "-m", self.whisper_model,
                        "-f", temp_wav,
                        "-t", "2",  # Use 2 threads
                        "--print-progress", "false",
                        "--print-special", "false"
                    ]

                    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
                    output, error = process.communicate()

                    if process.returncode == 0:
                        transcribed_text = output.decode().strip()
                        if transcribed_text:
                            print(f"Transcribed: {transcribed_text}")
                            response_text = self.generate_response(transcribed_text)
                            if response_text:
                                self.text_to_speech(response_text)
                            else:
                                messagebox.showerror("Error", "Could not generate response")
                        else:
                            messagebox.showinfo("Info", "No speech detected")
                    else:
                        print(f"Whisper error: {error.decode()}")
                        messagebox.showerror("Error", "Speech recognition failed")

                except Exception as e:
                    print(f"Processing error: {e}")
                    messagebox.showerror("Error", "Failed to process audio")

                finally:
                    # Cleanup
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)

    def text_to_speech(self, text: str) -> None:
        """Convert text to speech using either pyttsx3 or espeak"""
        if self.use_espeak:
            try:
                # Use espeak directly
                os.system(f'espeak "{text}"')
            except Exception as e:
                print(f"Error with espeak: {e}")
        else:
            try:
                temp_file = "temp/speech.wav"
                self.engine.save_to_file(text, temp_file)
                self.engine.runAndWait()
                
                # Use simpler playback method for Raspberry Pi
                os.system(f"aplay {temp_file}")
                
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error with TTS: {e}")
                # Fallback to espeak
                os.system(f'espeak "{text}"')

    def generate_response(self, user_input):
        """Generate response with minimal prompt"""
        system_prompt = """You are Immy, a friendly teddy bear. Be brief and simple."""

        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_input} [/INST]"""

        try:
            response = self.llm(
                prompt,
                max_tokens=128,  # Reduced max tokens
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1,
                top_k=40
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def create_gui(self):
        """Create a simple GUI"""
        self.window = tk.Tk()
        self.window.title("Immy - AI Teddy Bear")
        
        # Optimize window size for Raspberry Pi display
        self.window.geometry("320x240")

        frame = tk.Frame(self.window, padx=10, pady=10)
        frame.pack(expand=True)

        title_label = tk.Label(
            frame, 
            text="Talk with Immy",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)

        self.record_button = tk.Button(
            frame,
            text="Start Talking",
            font=("Arial", 12),
            padx=10,
            pady=5,
            bg="#4CAF50",
            fg="white"
        )
        self.record_button.pack(pady=10)

        def toggle_recording():
            if not self.recording:
                self.start_recording()
            else:
                self.stop_recording()

        self.record_button.config(command=toggle_recording)

        # Status label
        self.status_label = tk.Label(
            frame,
            text="Ready",
            font=("Arial", 10),
            fg="#666666"
        )
        self.status_label.pack(pady=5)

        self.window.mainloop()

if __name__ == "__main__":
    bot = SpeechBot()
    bot.create_gui()