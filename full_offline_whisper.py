import os
import time
from typing import IO
from io import BytesIO
import tkinter as tk
from tkinter import messagebox, ttk
import json
import sounddevice as sd
import numpy as np
import wave
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3
from llama_cpp import Llama
from whispercpp import Whisper

class SpeechBot:
    def __init__(self, 
                 llama_model_path="models/llama-2-7b-chat.gguf",
                 whisper_model_path="models/ggml-base.en.bin"):
        # Initialize llama.cpp
        self.llm = Llama(
            model_path=llama_model_path,
            n_ctx=2048,
            n_threads=4
        )

        # Initialize whisper.cpp
        self.whisper = Whisper(whisper_model_path)

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        
        # Default voice settings
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        # Audio recording settings
        self.sample_rate = 16000
        self.recording = False
        self.audio_data = []
        
        # Get available voices
        self.voices = self.engine.getProperty('voices')
        self.current_voice_idx = 0
        if self.voices:
            self.engine.setProperty('voice', self.voices[0].id)

    def set_voice_properties(self, rate, volume, voice_idx):
        """Update voice properties"""
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
        
        # Update button text
        self.record_button.config(text="Stop Talking", bg="#FF4444")
        
        # Start recording audio
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self.audio_callback
        )
        self.stream.start()

    def stop_recording(self):
        """Stop audio recording and process the audio"""
        self.recording = False
        self.stream.stop()
        self.stream.close()
        
        # Update button text
        self.record_button.config(text="Start Talking", bg="#4CAF50")
        
        # Process recorded audio
        if len(self.audio_data) > 0:
            # Convert audio data to numpy array
            audio_data = np.concatenate(self.audio_data)
            
            # Save as WAV file temporarily
            temp_wav = "temp_recording.wav"
            with wave.open(temp_wav, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            try:
                # Transcribe using whisper.cpp
                result = self.whisper.transcribe(temp_wav)
                transcribed_text = result
                
                if transcribed_text:
                    print(f"Transcribed: {transcribed_text}")
                    
                    # Generate and speak response
                    response_text = self.generate_response(transcribed_text)
                    print("LLaMA response:", response_text)
                    if response_text:
                        self.text_to_speech(response_text)
                    else:
                        print("No response generated.")
                        messagebox.showerror("Error", "Failed to generate response")
                else:
                    print("No speech detected")
                    messagebox.showinfo("Info", "No speech detected. Please try again!")

            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                messagebox.showerror("Error", "Failed to process speech")
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

    def text_to_speech(self, text: str) -> None:
        start_time = time.time()
        temp_file = "temp_speech.wav"
        self.engine.save_to_file(text, temp_file)
        self.engine.runAndWait()

        try:
            audio = AudioSegment.from_wav(temp_file)
            # Add some fun effects to make it more kid-friendly
            audio = audio + 3  # Slightly increase volume
            audio = audio.high_pass_filter(300)  # Reduce low frequencies
            play(audio)
        except Exception as e:
            print(f"Error playing audio: {str(e)}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        end_time = time.time()
        print(f"Text-to-speech conversion and playback took {end_time - start_time:.2f} seconds")

    def generate_response(self, user_input):
        system_prompt = """You are Immy, a magical AI-powered teddy bear who loves to chat with children. 
        You are kind, funny, and full of wonder, always ready to tell stories, answer questions, 
        and offer friendly advice. When speaking, you are playful, patient, and use simple, 
        child-friendly language. You encourage curiosity, learning, and imagination. 
        Keep your responses relatively brief and engaging."""

        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_input} [/INST]"""

        try:
            response = self.llm(
                prompt,
                max_tokens=256,
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1,
                top_k=40
            )

            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None

    def create_voice_settings_window(self):
        """Create a window for voice settings"""
        settings = tk.Toplevel(self.window)
        settings.title("Voice Settings")
        settings.geometry("300x400")

        # Voice selection
        tk.Label(settings, text="Select Voice:", font=("Arial", 12)).pack(pady=10)
        voice_var = tk.StringVar(value=self.voices[self.current_voice_idx].name)
        voice_combo = ttk.Combobox(settings, textvariable=voice_var)
        voice_combo['values'] = [voice.name for voice in self.voices]
        voice_combo.pack(pady=5)

        # Speed control
        tk.Label(settings, text="Speech Speed:", font=("Arial", 12)).pack(pady=10)
        speed_var = tk.IntVar(value=self.engine.getProperty('rate'))
        speed_scale = ttk.Scale(settings, from_=50, to=250, variable=speed_var, orient='horizontal')
        speed_scale.pack(pady=5)

        # Volume control
        tk.Label(settings, text="Volume:", font=("Arial", 12)).pack(pady=10)
        volume_var = tk.DoubleVar(value=self.engine.getProperty('volume'))
        volume_scale = ttk.Scale(settings, from_=0, to=1, variable=volume_var, orient='horizontal')
        volume_scale.pack(pady=5)

        def test_voice():
            voice_idx = voice_combo.current()
            self.set_voice_properties(speed_var.get(), volume_var.get(), voice_idx)
            self.text_to_speech("Hi! I'm Immy, your magical teddy bear friend!")

        ttk.Button(settings, text="Test Voice", command=test_voice).pack(pady=20)

        def save_settings():
            voice_idx = voice_combo.current()
            self.set_voice_properties(speed_var.get(), volume_var.get(), voice_idx)
            settings.destroy()

        ttk.Button(settings, text="Save Settings", command=save_settings).pack(pady=10)

    def create_gui(self):
        self.window = tk.Tk()
        self.window.title("Immy - Your AI Teddy Bear Friend")
        self.window.geometry("400x400")

        frame = tk.Frame(self.window, padx=20, pady=20)
        frame.pack(expand=True)

        title_label = tk.Label(
            frame, 
            text="Talk with Immy",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=20)

        instructions = tk.Label(
            frame,
            text="Press the button and start speaking!\nImmy is excited to chat with you!",
            font=("Arial", 12),
            justify=tk.CENTER
        )
        instructions.pack(pady=20)

        # Add voice settings button
        settings_button = tk.Button(
            frame,
            text="Voice Settings",
            command=self.create_voice_settings_window,
            font=("Arial", 12),
            padx=10,
            pady=5,
            bg="#4CAF50",
            fg="white"
        )
        settings_button.pack(pady=10)

        self.record_button = tk.Button(
            frame,
            text="Start Talking",
            font=("Arial", 14),
            padx=20,
            pady=10,
            bg="#4CAF50",
            fg="white"
        )
        self.record_button.pack(pady=20)

        # Configure record button to toggle recording
        def toggle_recording():
            if not self.recording:
                self.start_recording()
            else:
                self.stop_recording()

        self.record_button.config(command=toggle_recording)

        self.window.mainloop()

if __name__ == "__main__":
    bot = SpeechBot()
    bot.create_gui()