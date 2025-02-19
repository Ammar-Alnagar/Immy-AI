from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
import base64
import os
from openai import OpenAI
import tempfile
from pydub import AudioSegment
import io
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import edge_tts

# Initialize FastAPI app
app = FastAPI()

class TranscriptionService:
    def __init__(self):
        self.api_key = self._validate_api_key()
        self.client = self._initialize_openai_client()
        self.session_transcriptions: Dict[int, List[str]] = {}
        self.executor = ThreadPoolExecutor(max_workers=5)

    def _validate_api_key(self) -> str:
        """Validate and return the OpenAI API key."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return api_key

    def _initialize_openai_client(self) -> OpenAI:
        """Initialize and test OpenAI client."""
        try:
            client = OpenAI(api_key=self.api_key)
            client.models.list()  # Test the API key
            return client
        except Exception as e:
            raise ValueError(f"Invalid OpenAI API key: {str(e)}")

    async def process_audio_data(self, audio_data: str) -> str:
        """Process base64 encoded audio data and save to temporary file."""
        try:
            decoded_audio = base64.b64decode(audio_data)
            print(f"Decoded audio data size: {len(decoded_audio)} bytes")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(decoded_audio)
                return temp_audio.name
        except Exception as e:
            raise RuntimeError(f"Error processing audio data: {str(e)}")

    async def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio file using OpenAI Whisper API."""
        def _transcribe():
            request_client = OpenAI(api_key=self.api_key)
            with open(audio_file_path, 'rb') as audio_file:
                return request_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

        try:
            transcript = await asyncio.get_event_loop().run_in_executor(
                self.executor, _transcribe
            )
            return transcript.text
        except Exception as e:
            raise RuntimeError(f"Transcription error: {str(e)}")
        finally:
            try:
                os.unlink(audio_file_path)
                print("Cleaned up temporary files")
            except Exception as e:
                print(f"Error cleaning up temporary files: {e}")

class TTSService:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.voice = "en-US-JennyNeural"  # Default voice
        self.rate = "+0%"
        self.pitch = "+0Hz"

    async def text_to_speech(self, text: str) -> str:
        """Convert text to speech using Edge TTS."""
        tmp_path = None
        try:
            # Generate unique filename without creating the file
            timestamp = time.time_ns()
            tmp_path = os.path.join(tempfile.gettempdir(), f"tts_{timestamp}.mp3")

            # Initialize TTS and save directly to the path
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
            await communicate.save(tmp_path)
            
            # Read file in binary mode and encode to base64
            with open(tmp_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                return base64.b64encode(audio_data).decode('utf-8')

        except Exception as e:
            raise RuntimeError(f"TTS error: {str(e)}")
        finally:
            # Clean up temp file if it exists
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    print(f"Warning: Failed to clean up temp file {tmp_path}: {e}")

class LLMService:
    def __init__(self):
        self.api_key = self._validate_api_key()
        self.client = self._initialize_openai_client()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.role = "You are a helpful AI assistant that provides clear and concise answers."
    
    def _validate_api_key(self) -> str:
        """Validate and return the OpenAI API key."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return api_key

    def _initialize_openai_client(self) -> OpenAI:
        """Initialize and test OpenAI client."""
        try:
            client = OpenAI(api_key=self.api_key)
            client.models.list()  # Test the API key
            return client
        except Exception as e:
            raise ValueError(f"Invalid OpenAI API key: {str(e)}")
        
    async def answer(self, question: str) -> str:
        """Answer question using OpenAI LLM API."""
        def _answer():
            request_client = OpenAI(api_key=self.api_key)
            response = request_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.role},
                    {"role": "user", "content": question}
                ]
            )
            return response.choices[0].message.content
            
        try:
            answer = await asyncio.get_event_loop().run_in_executor(
                self.executor, _answer
            )
            return answer
        except Exception as e:
            raise RuntimeError(f"Answer error: {str(e)}")

class WebSocketManager:
    def __init__(self, transcription_service: TranscriptionService, llm_service: LLMService, tts_service: TTSService):
        self.transcription_service = transcription_service
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.pending_transcriptions: Dict[int, Dict[int, str]] = {}
        self.next_sequence: Dict[int, int] = {}

    async def handle_stop_signal(self, websocket: WebSocket, session_id: int) -> None:
        """Handle stop signal from client."""
        print(f"Stop signal received for session {session_id}")
        final_transcription = " ".join(
            self.transcription_service.session_transcriptions.get(session_id, [])
        )
        await websocket.send_json({"final_transcription": final_transcription})

    async def process_audio_chunk(
        self, 
        websocket: WebSocket, 
        session_id: int, 
        chunk_id: int,
        sequence: int,
        audio_data: str
    ) -> None:
        """Process a single audio chunk and send results to client."""
        chunk_start_time = time.perf_counter()
        print(f"Processing chunk {chunk_id} for session {session_id}")
        await websocket.send_json({"debug": "Processing audio data..."})

        try:
            # Process audio
            audio_process_start = time.perf_counter()
            temp_audio_path = await self.transcription_service.process_audio_data(audio_data)
            audio_process_end = time.perf_counter()
            audio_process_time = audio_process_end - audio_process_start

            # Transcribe audio
            await websocket.send_json({"debug": "Transcribing audio..."})
            transcribe_start = time.perf_counter()
            transcription_text = await self.transcription_service.transcribe_audio(temp_audio_path)
            transcribe_end = time.perf_counter()
            transcribe_time = transcribe_end - transcribe_start

            try:
                # Get LLM response
                await websocket.send_json({"debug": "Getting AI response..."})
                llm_start = time.perf_counter()
                llm_response = await self.llm_service.answer(transcription_text)
                llm_end = time.perf_counter()
                llm_time = llm_end - llm_start

                # Convert to speech
                await websocket.send_json({"debug": "Converting to speech..."})
                tts_start = time.perf_counter()
                tts_audio = await self.tts_service.text_to_speech(llm_response)
                tts_end = time.perf_counter()
                tts_time = tts_end - tts_start
            except Exception as e:
                raise RuntimeError(f"Processing error: {str(e)}")

            if transcription_text.strip():
                chunk_end_time = time.perf_counter()
                total_time = chunk_end_time - chunk_start_time

                # Store transcription
                if session_id not in self.pending_transcriptions:
                    self.pending_transcriptions[session_id] = {}
                    self.next_sequence[session_id] = 0
                
                self.pending_transcriptions[session_id][sequence] = transcription_text

                # Send timing information and responses
                await websocket.send_json({
                    "timing": {
                        "chunk_id": chunk_id,
                        "audio_process_time": round(audio_process_time * 1000, 2),
                        "transcribe_time": round(transcribe_time * 1000, 2),
                        "llm_time": round(llm_time * 1000, 2),
                        "tts_time": round(tts_time * 1000, 2),
                        "total_time": round(total_time * 1000, 2)
                    },
                    "transcription": transcription_text,
                    "llm_response": llm_response,
                    "tts_audio": tts_audio
                })

                # Process transcriptions in order
                await self._process_pending_transcriptions(websocket, session_id)

        except Exception as e:
            error_msg = f"Error processing audio chunk: {str(e)}"
            print(error_msg)
            await websocket.send_json({"error": error_msg})

    async def _process_pending_transcriptions(self, websocket: WebSocket, session_id: int) -> None:
        """Process pending transcriptions in sequence order."""
        while self.next_sequence[session_id] in self.pending_transcriptions[session_id]:
            text = self.pending_transcriptions[session_id][self.next_sequence[session_id]]
            if session_id not in self.transcription_service.session_transcriptions:
                self.transcription_service.session_transcriptions[session_id] = []
            
            self.transcription_service.session_transcriptions[session_id].append(text)
            await websocket.send_json({
                "transcription": text,
                "sequence": self.next_sequence[session_id]
            })
            
            del self.pending_transcriptions[session_id][self.next_sequence[session_id]]
            self.next_sequence[session_id] += 1

# Initialize services
transcription_service = TranscriptionService()
llm_service = LLMService()
tts_service = TTSService()
websocket_manager = WebSocketManager(transcription_service, llm_service, tts_service)

@app.get("/")
async def get():
    """Serve the main HTML page."""
    return FileResponse('index.html')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections and messages."""
    await websocket.accept()
    session_id = id(websocket)
    print(f"New WebSocket connection: {session_id}")
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                json_data = json.loads(data)
            except Exception as e:
                print(f"Error receiving WebSocket data: {str(e)}")
                await websocket.send_json({"error": "Connection error. Please try again."})
                break

            if json_data.get('stop'):
                await websocket_manager.handle_stop_signal(websocket, session_id)
                break

            if json_data.get('audio_data'):
                await websocket_manager.process_audio_chunk(
                    websocket,
                    session_id,
                    json_data.get('chunk_id'),
                    json_data.get('sequence', 0),
                    json_data['audio_data']
                )

    except Exception as e:
        error_details = f"WebSocket error: {str(e)}"
        print(error_details)
        try:
            await websocket.send_json({"error": error_details})
        except:
            print("Could not send error to client")
    finally:
        if session_id in transcription_service.session_transcriptions:
            del transcription_service.session_transcriptions[session_id]
        print(f"WebSocket connection closed: {session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
