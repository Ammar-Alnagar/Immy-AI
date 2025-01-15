import requests

# Your ElevenLabs API key
API_KEY = "sk_dee83966a5e3d57289bb6ed748776fb374cac26e29f931a4"

# ElevenLabs TTS endpoint
TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/jBpfuIE2acCO8z3wKNLl"

# Replace with the voice ID you want to use
VOICE_ID = "jBpfuIE2acCO8z3wKNLl"

# Text you want to convert to speech
TEXT ="""Hello there! It's such a pleasure to meet you today. My name is Alex, and I’m here to guide you through this wonderful journey. How can I assist you?

Wow, that’s incredible news! I’m so thrilled for you—this is truly a milestone worth celebrating.
But wait... oh no, that doesn’t sound good at all. I’m really sorry to hear about what happened. How can we make this better?
Haha, you’ve got to be kidding me! That’s absolutely hilarious—I can’t stop laughing!
Oh, no! Look out! That was a close call, wasn’t it? Phew!

The sun dipped below the horizon, painting the sky in shades of orange and pink. A gentle breeze rustled the leaves, carrying with it the sweet scent of blooming flowers. Somewhere in the distance, a river flowed, its waters sparkling like liquid silver under the moonlight.

So, here’s the thing—if you’re not entirely sure about it, why not take a step back and reconsider? I mean, there’s no harm in giving it another thought, right? It’s always better to be certain.

The quick brown fox jumps over the lazy dog. She sells seashells by the seashore. How much wood would a woodchuck chuck if a woodchuck could chuck wood? Peter Piper picked a peck of pickled peppers.

Why do we look at the stars and wonder about our place in the universe? What drives us to explore, to question, and to dream beyond the limits of what we know? Isn’t that what makes us human?

Stop right there! Listen carefully—this is something you cannot ignore. Take action immediately, and don’t let this opportunity slip away.

Let’s play a game! Are you ready? Here’s the first riddle: What has keys but can’t open locks? Got it? Okay, here’s another one!

In computational linguistics, phonemes are the smallest units of sound that distinguish one word from another in a specific language. For instance, in English, the words ‘bat’ and ‘pat’ differ by just one phoneme, the initial sound.

Under the silver moonlit sky,
Where whispers of the nightbirds fly,
The world feels vast, yet dreams are near,
A tranquil place, with nothing to fear.

Hmm, I’m not entirely sure about that. Let’s think this through carefully. Could there be another explanation we haven’t considered?

Can you believe it? This is unbelievable! What are the chances of something like this happening? Absolutely mind-blowing!

Take a deep breath. Close your eyes and let go of all the tension. Everything is going to be okay. Just focus on the present moment and let the calm wash over you.

Thank you so much for your time. It’s been wonderful talking with you. Until we meet again, take care, and stay safe!"""
# Output audio file
OUTPUT_FILE = "output_audio.mp3"

def text_to_speech(api_key, voice_id, text, output_file):
    headers = {
        "xi-api-key": 'sk_dee83966a5e3d57289bb6ed748776fb374cac26e29f931a4',
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.0,
            "similarity_boost": 1.0,
            "style": 0.2,
            "use_speaker_boost": True,
        }
    }

    # Send POST request to ElevenLabs TTS API
    response = requests.post(TTS_URL.format(voice_id=voice_id), json=payload, headers=headers)

    if response.status_code == 200:
        # Save the audio content to a file
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Audio has been saved to {output_file}")
    else:
        print(f"Failed to generate audio. Status code: {response.status_code}, Response: {response.text}")

# Call the function
text_to_speech(API_KEY, VOICE_ID, TEXT, OUTPUT_FILE)
