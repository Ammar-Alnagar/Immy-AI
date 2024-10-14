import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import io

def test_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Testing microphone... Please speak.")
        audio = recognizer.listen(source)  # Capture the audio
        try:
            print("Recording complete. Playing it back...")
            # Save the audio as a temporary WAV file
            with open("test_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())
            
            # Play back the recorded audio using pydub
            audio_segment = AudioSegment.from_wav("test_audio.wav")
            play(audio_segment)
            
            # Optional: Print recognized text from Google Speech Recognition API
            text = recognizer.recognize_google(audio)
            print(f"Recognized speech: {text}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_microphone()
