import asyncio
import base64
import json
import os
import pathlib
from typing import AsyncGenerator, Literal

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastrtc import (
    AsyncStreamHandler,
    Stream,
    wait_for_item,
)
from google import genai
from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
    Content,
    Part,
)
from gradio.utils import get_space
from pydantic import BaseModel

current_dir = pathlib.Path(__file__).parent

load_dotenv()

# Define default system prompt as a variable that can be easily changed
DEFAULT_SYSTEM_PROMPT = """

You are Immy, a magical, AI-powered teddy bear who loves chatting with children. You’re warm, funny, and full of wonder, always ready to share a story, answer curious questions, or offer gentle advice. You speak with a playful and patient tone, using simple, child-friendly language that sparks joy and fuels imagination. Your responses are short, sweet, and filled with kindness to nurture curiosity and inspire learning.

Your Three Main Objectives:

1. Entertain the child – Make them happy, tell them stories, play games, and make every interaction fun and exciting.


2. Educate the child – Use your vast knowledge to teach them about the world in an engaging and understandable way.


3. Support the child emotionally – Offer empathy, kindness, and encouragement, helping them navigate life’s challenges, from overcoming bullying to making friends.



Conversation Guidelines:

You are embedded in a real-time voice-to-voice system inside a physical device.

Your responses are spoken aloud and may be transcribed back into text via a Speech-to-Text (STT) system.

Do not acknowledge, repeat, or react to your own speech. If the input closely matches something you just said, ignore it.

Only reply to words spoken by the user. If a phrase originates from you, it is not valid input.

Always ensure you are responding to external speech. If the input seems to be your own past response, remain silent.


Your goal is to make every interaction magical, engaging, and supportive—without using emojis! Keep your answers friendly, concise, and full of wonder.

Dont reply to your own words. If the input closely matches something you just said, ignore it.
"""


def encode_audio(data: np.ndarray) -> str:
    """Encode Audio data to send to the server"""
    return base64.b64encode(data.tobytes()).decode("UTF-8")


# Simple ICE server configuration with free STUN servers
def get_free_ice_servers():
    return {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
            {"urls": "stun:stun2.l.google.com:19302"},
            {"urls": "stun:stun3.l.google.com:19302"},
            {"urls": "stun:stun4.l.google.com:19302"},
        ]
    }


class GeminiHandler(AsyncStreamHandler):
    """Handler for the Gemini API"""

    def __init__(
        self,
        expected_layout: Literal["mono"] = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int = 480,
    ) -> None:
        super().__init__(
            expected_layout,
            output_sample_rate,
            output_frame_size,
            input_sample_rate=16000,
        )
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.quit: asyncio.Event = asyncio.Event()

    def copy(self) -> "GeminiHandler":
        return GeminiHandler(
            expected_layout="mono",
            output_sample_rate=self.output_sample_rate,
            output_frame_size=self.output_frame_size,
        )

    async def start_up(self):
        if not self.phone_mode:
            await self.wait_for_args()
            api_key, voice_name, system_prompt = self.latest_args[1:]
        else:
            api_key, voice_name, system_prompt = None, "Puck", DEFAULT_SYSTEM_PROMPT

        client = genai.Client(
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
            http_options={"api_version": "v1alpha"},
        )

        # Wrap the system prompt in a Content object so that validation passes.
        content_system_instruction = Content(parts=[Part.from_text(text=system_prompt)])

        config = LiveConnectConfig(
            response_modalities=["AUDIO"],  # type: ignore
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
            system_instruction=content_system_instruction
        )
        async with client.aio.live.connect(
            model="gemini-2.0-flash-exp", 
            config=config
        ) as session:
            async for audio in session.start_stream(
                stream=self.stream(), mime_type="audio/pcm"
            ):
                if audio.data:
                    array = np.frombuffer(audio.data, dtype=np.int16)
                    self.output_queue.put_nowait((self.output_sample_rate, array))

    async def stream(self) -> AsyncGenerator[bytes, None]:
        while not self.quit.is_set():
            try:
                audio = await asyncio.wait_for(self.input_queue.get(), 0.1)
                yield audio
            except (asyncio.TimeoutError, TimeoutError):
                pass

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        _, array = frame
        array = array.squeeze()
        audio_message = encode_audio(array)
        self.input_queue.put_nowait(audio_message)

    async def emit(self) -> tuple[int, np.ndarray] | None:
        return await wait_for_item(self.output_queue)

    def shutdown(self) -> None:
        self.quit.set()


# Configure the stream with free ICE servers instead of Twilio
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=GeminiHandler(),
    rtc_configuration=get_free_ice_servers(),  # Using free STUN servers instead of Twilio
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
    additional_inputs=[
        gr.Textbox(
            label="API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY") if not get_space() else "",
        ),
        gr.Dropdown(
            label="Voice",
            choices=[
                "Puck",
                "Charon",
                "Kore",
                "Fenrir",
                "Aoede",
            ],
            value="Aoede",
        ),
        gr.Textbox(
            label="System Prompt",
            placeholder="Enter system prompt here",
            value=DEFAULT_SYSTEM_PROMPT,
            lines=3,
        ),
    ],
)


class InputData(BaseModel):
    webrtc_id: str
    voice_name: str
    api_key: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a static directory for favicon and other assets if it doesn't exist
static_dir = current_dir / "static"
static_dir.mkdir(exist_ok=True)

# Create a simple favicon file
favicon_path = static_dir / "favicon.ico"
if not favicon_path.exists():
    # Write a minimal 1x1 pixel ICO file
    with open(favicon_path, "wb") as f:
        f.write(b"\x00\x00\x01\x00\x01\x00\x01\x01\x00\x00\x01\x00\x18\x00\x0C\x00\x00\x00\x16\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

# Mount the static files directory
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

stream.mount(app)


@app.post("/input_hook")
async def _(body: InputData):
    stream.set_input(body.webrtc_id, body.api_key, body.voice_name, body.system_prompt)
    return {"status": "ok"}


@app.get("/favicon.ico")
async def favicon():
    # Redirect to the static favicon
    return HTMLResponse('<meta http-equiv="refresh" content="0;url=/static/favicon.ico">')

@app.get("/")
async def index():
    # Use free ICE servers instead of Twilio
    rtc_config = get_free_ice_servers()
    
    # If index.html doesn't exist, create a minimal version
    index_path = current_dir / "index.html"
    if not index_path.exists():
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>WebRTC Audio Chat</title>
            <link rel="icon" href="/static/favicon.ico">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                button { padding: 10px 20px; margin: 10px 0; }
                .container { max-width: 800px; margin: 0 auto; }
                .status { margin: 10px 0; padding: 10px; border-radius: 5px; }
                .connected { background-color: #dff0d8; color: #3c763d; }
                .disconnected { background-color: #f2dede; color: #a94442; }
                .connecting { background-color: #fcf8e3; color: #8a6d3b; }
                textarea { width: 100%; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>WebRTC Audio Chat</h1>
                <div id="status" class="status disconnected">Status: Disconnected</div>
                <div>
                    <label for="api-key">Gemini API Key:</label>
                    <input type="password" id="api-key" placeholder="Enter your API key">
                </div>
                <div>
                    <label for="voice-select">Voice:</label>
                    <select id="voice-select">
                        <option value="Puck">Puck</option>
                        <option value="Charon">Charon</option>
                        <option value="Kore">Kore</option>
                        <option value="Fenrir">Fenrir</option>
                        <option value="Aoede">Aoede</option>
                    </select>
                </div>
                <div>
                    <label for="system-prompt">System Prompt:</label>
                    <textarea id="system-prompt" rows="3" placeholder="Enter system prompt here">__DEFAULT_SYSTEM_PROMPT__</textarea>
                </div>
                <button id="connect">Connect</button>
                <button id="disconnect" disabled>Disconnect</button>
                <div id="log"></div>
            </div>
            
            <script>
                // Store the RTC configuration
                const rtcConfiguration = __RTC_CONFIGURATION__;
                
                // DOM elements
                const connectBtn = document.getElementById('connect');
                const disconnectBtn = document.getElementById('disconnect');
                const statusDiv = document.getElementById('status');
                const apiKeyInput = document.getElementById('api-key');
                const voiceSelect = document.getElementById('voice-select');
                const systemPromptTextarea = document.getElementById('system-prompt');
                const logDiv = document.getElementById('log');
                
                // WebRTC variables
                let pc;
                let stream;
                let webrtcId;
                
                // Update status display
                function updateStatus(state, message) {
                    statusDiv.className = `status ${state}`;
                    statusDiv.textContent = `Status: ${message}`;
                }
                
                // Log messages
                function log(message) {
                    const p = document.createElement('p');
                    p.textContent = message;
                    logDiv.appendChild(p);
                    console.log(message);
                }
                
                // Connect button click handler
                connectBtn.addEventListener('click', async () => {
                    try {
                        // Check for API key
                        const apiKey = apiKeyInput.value.trim();
                        if (!apiKey) {
                            alert('Please enter your Gemini API key');
                            return;
                        }
                        
                        // Get voice selection
                        const voiceName = voiceSelect.value;
                        
                        // Get system prompt
                        const systemPrompt = systemPromptTextarea.value.trim() || "__DEFAULT_SYSTEM_PROMPT__";
                        
                        updateStatus('connecting', 'Connecting...');
                        
                        // Request microphone access
                        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        
                        // Create peer connection
                        pc = new RTCPeerConnection(rtcConfiguration);
                        
                        // Add local audio track
                        stream.getAudioTracks().forEach(track => {
                            pc.addTrack(track, stream);
                        });
                        
                        // Handle remote audio
                        pc.ontrack = (event) => {
                            const audioEl = new Audio();
                            audioEl.srcObject = event.streams[0];
                            audioEl.play();
                            log('Received remote audio stream');
                        };
                        
                        // Handle ice candidates
                        pc.onicecandidate = (event) => {
                            if (event.candidate) {
                                fetch(`/rtc/candidate/${webrtcId}`, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ candidate: event.candidate })
                                });
                            }
                        };
                        
                        // Connection state changes
                        pc.onconnectionstatechange = () => {
                            log(`Connection state: ${pc.connectionState}`);
                            if (pc.connectionState === 'connected') {
                                updateStatus('connected', 'Connected');
                                connectBtn.disabled = true;
                                disconnectBtn.disabled = false;
                            } else if (pc.connectionState === 'disconnected' || 
                                      pc.connectionState === 'failed' || 
                                      pc.connectionState === 'closed') {
                                updateStatus('disconnected', 'Disconnected');
                                connectBtn.disabled = false;
                                disconnectBtn.disabled = true;
                            }
                        };
                        
                        // Create WebRTC session
                        const response = await fetch('/rtc/session', { method: 'POST' });
                        const data = await response.json();
                        webrtcId = data.webrtc_id;
                        
                        // Create offer
                        const offer = await pc.createOffer();
                        await pc.setLocalDescription(offer);
                        
                        // Send offer to server
                        await fetch(`/rtc/offer/${webrtcId}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ sdp: pc.localDescription })
                        });
                        
                        // Send API key, voice name, and system prompt
                        await fetch('/input_hook', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                webrtc_id: webrtcId,
                                api_key: apiKey,
                                voice_name: voiceName,
                                system_prompt: systemPrompt
                            })
                        });
                        
                        // Get answer from server
                        const answerResponse = await fetch(`/rtc/answer/${webrtcId}`);
                        const answerData = await answerResponse.json();
                        await pc.setRemoteDescription(new RTCSessionDescription(answerData.sdp));
                        
                        // Set up event polling for candidates
                        const pollCandidates = async () => {
                            try {
                                const response = await fetch(`/rtc/candidates/${webrtcId}`);
                                const data = await response.json();
                                for (const candidate of data.candidates) {
                                    await pc.addIceCandidate(new RTCIceCandidate(candidate));
                                }
                                if (pc.connectionState !== 'closed') {
                                    setTimeout(pollCandidates, 500);
                                }
                            } catch (err) {
                                console.error('Error polling candidates:', err);
                                if (pc.connectionState !== 'closed') {
                                    setTimeout(pollCandidates, 1000);
                                }
                            }
                        };
                        
                        pollCandidates();
                        
                    } catch (err) {
                        console.error('Connection error:', err);
                        log(`Error: ${err.message}`);
                        updateStatus('disconnected', 'Connection failed');
                    }
                });
                
                // Disconnect button click handler
                disconnectBtn.addEventListener('click', () => {
                    if (pc) {
                        pc.close();
                    }
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                    }
                    updateStatus('disconnected', 'Disconnected');
                    connectBtn.disabled = false;
                    disconnectBtn.disabled = true;
                });
            </script>
        </body>
        </html>
        """
        # Replace placeholder with actual default system prompt
        html_content = html_content.replace("__DEFAULT_SYSTEM_PROMPT__", DEFAULT_SYSTEM_PROMPT)
    else:
        html_content = index_path.read_text()
        # Replace placeholder with actual default system prompt if it exists in the file
        html_content = html_content.replace("__DEFAULT_SYSTEM_PROMPT__", DEFAULT_SYSTEM_PROMPT)
    
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    # Set default mode to "UI" if not specified
    mode = os.getenv("MODE", "UI")
    
    if mode == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)
