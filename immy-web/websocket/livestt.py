from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
import base64
import os
from openai import AsyncOpenAI
import tempfile
from pydub import AudioSegment
import io
import time
import asyncio
from typing import Dict, List, Optional, Set
import edge_tts
from collections import deque

# Initialize FastAPI app
app = FastAPI()

# Mount static files directory for serving JavaScript and other static files
app.mount("/static", StaticFiles(directory="."), name="static")

class TranscriptionService:
    def __init__(self):
        self.api_key = self._validate_api_key()
        self.client = self._initialize_openai_client()
        self.session_transcriptions: Dict[int, List[str]] = {}

    def _validate_api_key(self) -> str:
        """Validate and return the OpenAI API key."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return api_key

    def _initialize_openai_client(self) -> AsyncOpenAI:
        """Initialize and test OpenAI client."""
        try:
            client = AsyncOpenAI(api_key=self.api_key)
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
        try:
            with open(audio_file_path, 'rb') as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
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
        self.voice = "en-US-JennyNeural"
        self.rate = "+0%"
        self.pitch = "+0Hz"
        self.tts_queue = deque()
        self.processing = False
        self.current_chunk_id = None
        self.active_chunks: Set[int] = set()

    async def text_to_speech(self, text: str, chunk_id: int, sequence: int, metrics: dict) -> None:
        """Queue text for TTS processing."""
        if not text.strip():
            return

        # Add to queue
        self.tts_queue.append({
            'text': text,
            'chunk_id': chunk_id,
            'sequence': sequence,
            'metrics': metrics
        })

        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_tts_queue())

    async def _process_tts_queue(self) -> None:
        """Process TTS queue in order."""
        self.processing = True
        try:
            while self.tts_queue:
                item = self.tts_queue.popleft()
                if item['chunk_id'] not in self.active_chunks:
                    continue

                text = item['text']
                chunk_id = item['chunk_id']
                metrics = item['metrics']
                
                try:
                    metrics['ttsStart'] = round(time.perf_counter() * 1000)
                    
                    # Create audio data
                    audio_data = io.BytesIO()
                    try:
                        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
                        
                        async for chunk in communicate.stream():
                            if chunk["type"] == "audio":
                                if chunk_id not in self.active_chunks:
                                    break
                                audio_data.write(chunk["data"])
                        
                        if not audio_data.getvalue():
                            print(f"Warning: No audio data generated for text: {text}")
                            return
                        
                        if chunk_id in self.active_chunks:
                            # Get the audio data and encode to base64
                            audio_data.seek(0)
                            audio_base64 = base64.b64encode(audio_data.getvalue()).decode('utf-8')
                            
                            metrics['ttsEnd'] = round(time.perf_counter() * 1000)
                            
                            # Send TTS chunk to client
                            await websocket_manager.send_tts_chunk(
                                chunk_id,
                                item['sequence'],
                                audio_base64,
                                text,
                                metrics
                            )
                    except Exception as e:
                        print(f"Error generating TTS audio for text '{text}': {str(e)}")
                        return
                
                except Exception as e:
                    print(f"TTS error for chunk {chunk_id}: {str(e)}")
                    
        finally:
            self.processing = False

    def clear_queue(self, chunk_id: Optional[int] = None) -> None:
        """Clear TTS queue for specific chunk or all chunks."""
        if chunk_id is not None:
            self.active_chunks.discard(chunk_id)
            self.tts_queue = deque(item for item in self.tts_queue if item['chunk_id'] != chunk_id)
        else:
            self.active_chunks.clear()
            self.tts_queue.clear()

    def start_chunk(self, chunk_id: int) -> None:
        """Mark a chunk as active for processing."""
        self.active_chunks.add(chunk_id)

class LLMService:
    def __init__(self):
        self.api_key = self._validate_api_key()
        self.client = self._initialize_openai_client()
        self.role = "You are a helpful AI assistant that provides clear and concise answers."
        self.max_tokens = 250
        self.llm_queue = deque()
        self.processing = False
        self.sequence_counter = 0

    def _validate_api_key(self) -> str:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return api_key

    def _initialize_openai_client(self) -> AsyncOpenAI:
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            return client
        except Exception as e:
            raise ValueError(f"Invalid OpenAI API key: {str(e)}")

    async def stream_answer(self, question: str, chunk_id: int, metrics: dict) -> str:
        """Stream answer using OpenAI LLM API with TTS streaming."""
        try:
            metrics['llmStart'] = round(time.perf_counter() * 1000)
            tts_service.start_chunk(chunk_id)
            
            # Reset sequence counter for new chunk
            self.sequence_counter = 0
            
            stream = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.role},
                    {"role": "user", "content": question}
                ],
                max_tokens=self.max_tokens,
                stream=True
            )
            
            full_response = ""
            current_chunk = ""
            token_count = 0
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    current_chunk += content
                    token_count += 1
                    
                    # Send text chunk to client
                    await websocket_manager.send_llm_chunk(chunk_id, content, metrics)
                    
                    # When we have enough tokens or hit sentence boundary, generate TTS
                    # Using smaller chunks for better performance
                    if token_count >= 30 or (content in ['.', '!', '?'] and token_count > 20):
                        if current_chunk.strip():
                            self.sequence_counter += 1
                            await tts_service.text_to_speech(
                                current_chunk,
                                chunk_id,
                                self.sequence_counter,
                                metrics.copy()
                            )
                        current_chunk = ""
                        token_count = 0
            
            # Process any remaining text
            if current_chunk.strip():
                self.sequence_counter += 1
                await tts_service.text_to_speech(
                    current_chunk,
                    chunk_id,
                    self.sequence_counter,
                    metrics.copy()
                )
            
            metrics['llmEnd'] = round(time.perf_counter() * 1000)
            return full_response
        except Exception as e:
            tts_service.clear_queue(chunk_id)
            raise RuntimeError(f"Answer error: {str(e)}")

class WebSocketManager:
    def __init__(self, transcription_service: TranscriptionService, llm_service: LLMService, tts_service: TTSService):
        self.transcription_service = transcription_service
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.pending_transcriptions: Dict[int, Dict[int, str]] = {}
        self.next_sequence: Dict[int, int] = {}
        self.active_websockets: Dict[int, WebSocket] = {}

    async def register_websocket(self, websocket: WebSocket, session_id: int) -> None:
        """Register a new WebSocket connection."""
        self.active_websockets[session_id] = websocket

    async def unregister_websocket(self, session_id: int) -> None:
        """Unregister a WebSocket connection."""
        self.active_websockets.pop(session_id, None)
        if session_id in self.pending_transcriptions:
            del self.pending_transcriptions[session_id]
        if session_id in self.next_sequence:
            del self.next_sequence[session_id]

    async def send_llm_chunk(self, chunk_id: int, content: str, metrics: dict) -> None:
        """Send LLM chunk to all active connections."""
        message = {
            "type": "llm_chunk",
            "content": content,
            "chunkId": chunk_id,
            "metrics": metrics
        }
        await self._broadcast(message)

    async def send_tts_chunk(self, chunk_id: int, sequence: int, audio: str, text: str, metrics: dict) -> None:
        """Send TTS chunk to all active connections."""
        message = {
            "type": "tts_chunk",
            "audio": audio,
            "text": text,
            "chunkId": chunk_id,
            "sequence": sequence,
            "metrics": metrics
        }
        await self._broadcast(message)

    async def _broadcast(self, message: dict) -> None:
        """Broadcast message to all active connections."""
        # Create a list of websockets to remove
        to_remove = []
        
        for session_id, websocket in self.active_websockets.items():
            try:
                try:
                    # Try to ping the client to check connection
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    # If we can't send, consider it disconnected
                    to_remove.append(session_id)
                    continue
                    
                await websocket.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to session {session_id}: {str(e)}")
                to_remove.append(session_id)
        
        # Clean up disconnected websockets
        for session_id in to_remove:
            await self.unregister_websocket(session_id)
            print(f"Removed disconnected session: {session_id}")

    async def handle_stop_signal(self, session_id: int) -> None:
        """Handle stop signal from client."""
        try:
            # Clear TTS queue first
            tts_service.clear_queue()
            
            # Get websocket if it exists
            websocket = self.active_websockets.get(session_id)
            if websocket:
                    try:
                        await websocket.close(code=1000, reason="Client requested stop")
                    except Exception as close_error:
                        print(f"Error closing websocket for session {session_id}: {str(close_error)}")
            
            # Always unregister the session
            await self.unregister_websocket(session_id)
            print(f"Cleaned up session {session_id}")
        except Exception as e:
            print(f"Error during stop signal handling for session {session_id}: {str(e)}")
            # Ensure websocket is unregistered even if error occurs
            try:
                await self.unregister_websocket(session_id)
            except:
                pass

    async def process_audio_chunk(
        self, 
        session_id: int, 
        chunk_id: int,
        sequence: int,
        audio_data: str
    ) -> None:
        """Process a single audio chunk and send results to client."""
        chunk_start_time = time.perf_counter()
        print(f"Processing chunk {chunk_id} for session {session_id}")
        
        try:
            # Initialize metrics for this chunk
            # Initialize all metrics with default values
            chunk_metrics = {
                'processingStart': round(chunk_start_time * 1000),
                'transcriptionStart': round(time.perf_counter() * 1000),
                'transcriptionEnd': 0,
                'llmStart': 0,
                'llmEnd': 0,
                'ttsStart': 0,
                'ttsEnd': 0,
                'totalTime': 0
            }

            # Process and transcribe audio in parallel
            audio_process_task = asyncio.create_task(
                self.transcription_service.process_audio_data(audio_data)
            )
            temp_audio_path = await audio_process_task
            
            transcribe_task = asyncio.create_task(
                self.transcription_service.transcribe_audio(temp_audio_path)
            )
            transcription_text = await transcribe_task
            
            # Update metrics after transcription
            chunk_metrics['transcriptionEnd'] = round(time.perf_counter() * 1000)

            # Stream LLM response with integrated TTS
            llm_response = await self.llm_service.stream_answer(transcription_text, chunk_id, chunk_metrics)

            # Store transcription
            if session_id not in self.pending_transcriptions:
                self.pending_transcriptions[session_id] = {}
                self.next_sequence[session_id] = 0
            
            self.pending_transcriptions[session_id][sequence] = transcription_text

            # Update final metrics
            chunk_metrics['processingEnd'] = round(time.perf_counter() * 1000)
            chunk_metrics['totalTime'] = chunk_metrics['processingEnd'] - chunk_metrics['processingStart']

            # Send final completion signal
            await self._broadcast({
                "type": "final_response",
                "transcription": transcription_text,
                "chunkId": chunk_id,
                "metrics": chunk_metrics
            })

        except Exception as e:
            error_msg = f"Error processing audio chunk: {str(e)}"
            print(error_msg)
            tts_service.clear_queue(chunk_id)
            await self._broadcast({"error": error_msg})

# Initialize services
transcription_service = TranscriptionService()
llm_service = LLMService()
tts_service = TTSService()
websocket_manager = WebSocketManager(transcription_service, llm_service, tts_service)

@app.get("/")
async def get():
    """Serve the main HTML page."""
    return FileResponse('./index.html')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections and messages."""
    session_id = id(websocket)
    print(f"New WebSocket connection request: {session_id}")
    
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted: {session_id}")
        
        # Register websocket with manager
        await websocket_manager.register_websocket(websocket, session_id)
        
        while True:
            try:
                # Check if connection is still alive
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    print(f"Client disconnected for session {session_id}")
                    break
                    
                data = await websocket.receive_text()
                json_data = json.loads(data)
                
                if json_data.get('stop'):
                    print(f"Stop signal received for session {session_id}")
                    await websocket_manager.handle_stop_signal(session_id)
                    break

                if json_data.get('audio_data'):
                    await websocket_manager.process_audio_chunk(
                        session_id,
                        json_data.get('chunk_id'),
                        json_data.get('sequence', 0),
                        json_data['audio_data']
                    )

            except json.JSONDecodeError as e:
                print(f"Invalid JSON received from session {session_id}: {str(e)}")
                try:
                    await websocket.send_json({"error": "Invalid message format"})
                except Exception:
                    print(f"Could not send error to client {session_id}")
                break
                
            except Exception as e:
                print(f"Error processing message for session {session_id}: {str(e)}")
                try:
                    await websocket.send_json({"error": "Connection error"})
                except Exception:
                    print(f"Could not send error to client {session_id}")
                break

    except Exception as e:
        print(f"WebSocket connection error for session {session_id}: {str(e)}")
    
    finally:
        # Always clean up
        try:
            await websocket_manager.handle_stop_signal(session_id)
        except Exception as e:
            print(f"Error during cleanup for session {session_id}: {str(e)}")
            # Last resort cleanup
            try:
                await websocket_manager.unregister_websocket(session_id)
            except:
                pass
        
        print(f"WebSocket connection closed and cleaned up: {session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
