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

app = FastAPI()

# Initialize OpenAI client with validation
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

try:
    client = OpenAI(api_key=api_key)
    # Test the API key with a simple request
    client.models.list()
except Exception as e:
    raise ValueError(f"Invalid OpenAI API key: {str(e)}")

# Store transcriptions for each session
session_transcriptions = {}

@app.get("/")
async def get():
    return FileResponse('index.html')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = id(websocket)
    session_transcriptions[session_id] = []
    print(f"New WebSocket connection: {session_id}")
    
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Store pending transcriptions with their sequence numbers
    pending_transcriptions = {}
    next_sequence = 0
    executor = ThreadPoolExecutor(max_workers=5)
    
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
                print(f"Stop signal received for session {session_id}")
                final_transcription = " ".join(session_transcriptions[session_id])
                await websocket.send_json({
                    "final_transcription": final_transcription
                })
                break
                
            if json_data.get('audio_data'):
                try:
                    chunk_start_time = time.perf_counter()
                    print(f"Processing chunk {json_data.get('chunk_id')} for session {session_id}")
                    await websocket.send_json({"debug": "Processing audio data..."})
                    
                    # Timing for audio processing
                    audio_process_start = time.perf_counter()
                    # Decode base64 audio
                    audio_data = base64.b64decode(json_data['audio_data'])
                    print(f"Decoded audio data size: {len(audio_data)} bytes")
                    
                    # Create temporary files for audio processing
                    audio_process_end = time.perf_counter()
                    audio_process_time = audio_process_end - audio_process_start
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                        temp_audio.write(audio_data)
                        temp_audio_path = temp_audio.name
                    
                    try:
                        await websocket.send_json({"debug": "Transcribing audio..."})
                        
                        # Timing for API call
                        api_call_start = time.perf_counter()
                        
                        # Create a new client instance for each request to ensure independence
                        def transcribe_audio():
                            # Create a new client for this request
                            request_client = OpenAI(api_key=client.api_key)
                            with open(temp_audio_path, 'rb') as audio_file:
                                return request_client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=audio_file
                                )
                        
                        # Run transcription in thread pool
                        transcript = await asyncio.get_event_loop().run_in_executor(
                            executor, transcribe_audio
                        )
                        
                        api_call_end = time.perf_counter()
                        api_call_time = api_call_end - api_call_start
                        
                        if transcript.text.strip():
                            chunk_end_time = time.perf_counter()
                            total_time = chunk_end_time - chunk_start_time
                            sequence = json_data.get('sequence', 0)
                            print(f"Transcription received for chunk {sequence}: {transcript.text}")
                            
                            # Store in pending transcriptions
                            pending_transcriptions[sequence] = transcript.text
                            
                            # Send timing information
                            await websocket.send_json({
                                "timing": {
                                    "chunk_id": json_data.get('chunk_id'),
                                    "audio_process_time": round(audio_process_time * 1000, 2),  # Convert to ms
                                    "api_call_time": round(api_call_time * 1000, 2),  # Convert to ms
                                    "total_time": round(total_time * 1000, 2)  # Convert to ms
                                }
                            })
                            
                            # Process transcriptions in order
                            while next_sequence in pending_transcriptions:
                                text = pending_transcriptions[next_sequence]
                                session_transcriptions[session_id].append(text)
                                await websocket.send_json({
                                    "transcription": text,
                                    "sequence": next_sequence
                                })
                                del pending_transcriptions[next_sequence]
                                next_sequence += 1
                        
                    except Exception as e:
                        error_msg = f"Error in transcription: {str(e)}"
                        print(error_msg)
                        await websocket.send_json({"error": error_msg})
                    
                    finally:
                        # Clean up temporary files
                        try:
                            os.unlink(temp_audio_path)
                            print("Cleaned up temporary files")
                        except Exception as e:
                            print(f"Error cleaning up temporary files: {e}")
                
                except Exception as e:
                    error_msg = f"Error processing audio: {str(e)}"
                    print(error_msg)
                    await websocket.send_json({"error": error_msg})
                    
    except Exception as e:
        error_details = f"WebSocket error: {str(e)}"
        print(error_details)
        try:
            await websocket.send_json({"error": error_details})
        except:
            print("Could not send error to client")
    finally:
        if session_id in session_transcriptions:
            del session_transcriptions[session_id]
        print(f"WebSocket connection closed: {session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
