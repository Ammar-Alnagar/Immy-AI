# Immy the AI Teddy Bear

This project brings to life Immy, an AI-powered teddy bear companion designed for delightful conversations with children. Immy uses speech recognition, natural language processing with OpenAI, and text-to-speech to interact in a playful and engaging way.

## Features

* **Engaging Conversations:** Immy provides short, sweet, and kind responses tailored for children, fostering curiosity and learning.
* **Wake/Sleep Functionality:** Immy can be "awakened" with the phrase "hey teddy" and put to "sleep" with "good night."
* **Child-Friendly Language:** Immy uses simple vocabulary and a playful tone, making interactions enjoyable for young users.
* **Filler Words:** Immy uses filler words (e.g., "um," "uh") before responding, creating a more natural conversational flow.  These are pre-cached for faster response times.
* **WebSocket Integration:**  Allows for external applications to receive Immy's responses in real-time. (See "Running with WebSocket Support" below).
* **Streaming Responses:** Immy streams responses from OpenAI, making the conversation feel more dynamic and less delayed.
* **Optimized TTS:** Uses Edge-TTS for high-quality, expressive speech.  TTS conversion is performed in a separate thread for smoother operation.
* **Asynchronous Operations:** Leverages `asyncio` for efficient handling of TTS and WebSocket communication.
* **Clear System Prompt:** The system prompt clearly defines Immy's persona and behavior.

## Technologies Used

* **Python:** The core programming language.
* **OpenAI API:** For natural language processing and conversation generation (gpt-4o-mini is the recommended model).
* **Edge-TTS:** For text-to-speech conversion.
* **SpeechRecognition:** For voice input.
* **SoundDevice:** For audio playback.
* **PyDub:** For audio file manipulation.
* **NumPy:** For numerical operations on audio data.
* **WebSockets:** For real-time communication with external applications.
* **python-dotenv:** For managing environment variables (API keys).

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/ImmyTheAITeddyBear.git](https://www.google.com/search?q=https://github.com/YOUR_USERNAME/ImmyTheAITeddyBear.git)  # Replace with your repo URL
   cd ImmyTheAITeddyBear

 * Create a virtual environment (recommended):
   python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

 * Install the required packages:
   pip install -r requirements.txt

 * Set up environment variables:
   * Create a .env file in the project directory.
   * Add your OpenAI API key:
     OPENAI_API_KEY=your_openai_api_key_here

Running the Application
 * Run the main script:
   python immy.py

 * Interact with Immy: Say "hey teddy" to wake Immy up, and "good night" to put Immy to sleep.  Then have a conversation!
Running with WebSocket Support
The application includes a WebSocket server that broadcasts Immy's responses.  This allows you to create separate applications that can listen to Immy's output in real-time.
 * Start Immy: Run python immy.py as usual.  The WebSocket server will start in the background.
 * Connect a WebSocket client: You can use any WebSocket client to connect to ws://localhost:8765.  When Immy speaks, the responses will be sent as JSON messages (e.g., {"message": "Hello there!"}).  You can use a simple JavaScript client in a browser or any other WebSocket library.
Project Structure
ImmyTheAITeddyBear/
├── immy.py          # Main script
├── requirements.txt # Project dependencies
├── .env             # Environment variables (API keys)
└── ...              # Other files (READMe)
