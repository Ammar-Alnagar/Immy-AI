import asyncio
import json
import websockets
from datetime import datetime
from services.transcription import TranscriptionService
from services.llm import LLMService
from services.tts import TTSService
from services.logger import Logger

class AIServer:
    def __init__(self):
        self.logger = Logger("AIServer")
        self.logger.info("Initializing AI Server")
        
        # Initialize services
        self.transcription_service = TranscriptionService()
        self.llm_service = LLMService()
        self.tts_service = TTSService()
        self.data_channel = None
        self.peer_connection = None
        
        self.logger.info("AI Server initialized successfully")

    async def handle_audio_data(self, message: dict):
        """Process audio data through transcription, LLM, and TTS pipeline."""
        start_time = datetime.now().timestamp()
        chunk_id = message.get('chunk_id')
        sequence = message.get('sequence')
        
        try:
            # Process metrics
            metrics = {
                'processingStart': message.get('timestamp', 0),
                'transcriptionStart': 0,
                'transcriptionEnd': 0,
                'llmStart': 0,
                'llmEnd': 0,
                'ttsStart': 0,
                'ttsEnd': 0
            }

            # Get audio data
            audio_data = message.get('audio_data')
            
            if not audio_data or not chunk_id:
                self.logger.error("Invalid message format", {
                    "has_audio": bool(audio_data),
                    "has_chunk_id": bool(chunk_id),
                    "sequence": sequence
                })
                return

            self.logger.info("Received audio chunk for processing", {
                "chunk_id": chunk_id,
                "sequence": sequence,
                "audio_size": len(audio_data),
                "timestamp": message.get('timestamp')
            })

            # Process and transcribe audio
            self.logger.debug("Starting audio data processing", {
                "chunk_id": chunk_id,
                "sequence": sequence
            })
            audio_file = await self.transcription_service.process_audio_data(audio_data)
            
            self.logger.debug("Starting transcription", {
                "chunk_id": chunk_id,
                "sequence": sequence,
                "audio_file": audio_file
            })
            
            metrics['transcriptionStart'] = message.get('timestamp', 0)
            transcription = await self.transcription_service.transcribe_audio(audio_file)
            metrics['transcriptionEnd'] = message.get('timestamp', 0)
            
            transcription_duration = metrics['transcriptionEnd'] - metrics['transcriptionStart']
            self.logger.info("Transcription completed", {
                "chunk_id": chunk_id,
                "sequence": sequence,
                "text_length": len(transcription) if transcription else 0,
                "duration_ms": transcription_duration
            })

            if not transcription or transcription.strip() == "":
                self.logger.warning("Empty transcription result", {
                    "chunk_id": chunk_id,
                    "sequence": sequence,
                    "audio_size": len(audio_data)
                })
                return

            # Start TTS chunk processing
            self.tts_service.start_chunk(chunk_id)
            self.logger.debug("Started TTS chunk processing", {
                "chunk_id": chunk_id,
                "sequence": sequence
            })
        
            # Stream LLM response
            self.logger.info("Starting LLM response streaming", {
                "chunk_id": chunk_id,
                "sequence": sequence,
                "input_length": len(transcription)
            })
            
            metrics['llmStart'] = message.get('timestamp', 0)
            llm_sequence = 0
            async for text_chunk in self.llm_service.stream_answer(transcription, metrics):
                llm_sequence += 1
                self.logger.debug("Processing LLM chunk", {
                    "chunk_id": chunk_id,
                    "sequence": sequence,
                    "llm_sequence": llm_sequence,
                    "chunk_length": len(text_chunk)
                })
                
                # Convert text chunk to speech
                metrics['ttsStart'] = message.get('timestamp', 0)
                self.logger.debug("Starting TTS conversion", {
                    "chunk_id": chunk_id,
                    "sequence": sequence,
                    "llm_sequence": llm_sequence,
                    "text_length": len(text_chunk)
                })
                
                audio_base64 = await self.tts_service.text_to_speech(
                    text_chunk,
                    chunk_id,
                    llm_sequence,
                    metrics
                )
                metrics['ttsEnd'] = message.get('timestamp', 0)
                
                tts_duration = metrics['ttsEnd'] - metrics['ttsStart']
                if audio_base64:
                    self.logger.info("Sending response to client", {
                        "chunk_id": chunk_id,
                        "sequence": sequence,
                        "llm_sequence": llm_sequence,
                        "text_length": len(text_chunk),
                        "audio_size": len(audio_base64),
                        "tts_duration_ms": tts_duration
                    })
                    # Send response through websocket
                    # Ensure audio is in correct format
                    if audio_base64:
                        response = {
                            'type': 'response',
                            'text': text_chunk,
                            'audio': audio_base64,
                            'chunkId': chunk_id,
                            'sequence': llm_sequence,
                            'metrics': metrics
                        }
                        
                        # Log response details before sending
                        self.logger.debug("Preparing response", {
                            "chunk_id": chunk_id,
                            "sequence": llm_sequence,
                            "text_length": len(text_chunk),
                            "audio_size": len(audio_base64),
                            "audio_format": "wav"
                        })
                        
                        # Send response through websocket
                        await self.websocket.send(json.dumps(response))
                        
                        # Log after sending
                        self.logger.debug("Response sent successfully", {
                            "chunk_id": chunk_id,
                            "sequence": llm_sequence,
                            "audio_sent": True
                        })
                    else:
                        self.logger.warning("No audio data in response", {
                            "chunk_id": chunk_id,
                            "sequence": llm_sequence,
                            "text_length": len(text_chunk)
                        })
                else:
                    self.logger.warning("Failed to generate audio", {
                        "chunk_id": chunk_id,
                        "sequence": sequence,
                        "llm_sequence": llm_sequence,
                        "text_length": len(text_chunk),
                        "tts_duration_ms": tts_duration
                    })

            metrics['llmEnd'] = message.get('timestamp', 0)
            llm_duration = metrics['llmEnd'] - metrics['llmStart']

            # Calculate total processing time
            total_time = round((datetime.now().timestamp() - start_time) * 1000, 2)
            metrics['totalTime'] = total_time

            # Send final transcription
            final_response = {
                'type': 'final',
                'transcription': transcription,
                'chunkId': chunk_id,
                'metrics': metrics
            }
            await self.websocket.send(json.dumps(final_response))
            
            self.logger.info("Completed full pipeline processing", {
                "chunk_id": chunk_id,
                "sequence": sequence,
                "total_time_ms": total_time,
                "transcription_length": len(transcription),
                "llm_duration_ms": llm_duration,
                "llm_chunks": llm_sequence
            })

        except Exception as e:
            self.logger.error("Error processing audio", {
                "chunk_id": chunk_id,
                "error": str(e)
            }, exc_info=True)
            error_message = {
                'type': 'error',
                'message': str(e),
                'chunkId': chunk_id
            }
            await self.websocket.send(json.dumps(error_message))
        finally:
            if chunk_id:
                self.tts_service.clear_chunk(chunk_id)

    async def connect_to_signaling_server(self, room_id: str):
        """Connect to signaling server and handle WebRTC signaling."""
        uri = f"ws://localhost:8000/ws/server/{room_id}"
        max_retries = 3
        retry_count = 0
        retry_delay = 1  # Start with 1 second delay
        
        while True:  # Keep trying to maintain connection
            try:
                self.logger.info("Connecting to signaling server", {
                    "room_id": room_id,
                    "uri": uri,
                    "attempt": retry_count + 1,
                    "total_retries": retry_count
                })
                
                async with websockets.connect(uri) as websocket:
                    self.websocket = websocket
                    retry_count = 0  # Reset retry count on successful connection
                    retry_delay = 1  # Reset delay
                    
                    self.logger.info("Connected to signaling server", {
                        "room_id": room_id,
                        "websocket_id": id(websocket)
                    })
                    
                    try:
                        async for message in websocket:
                            try:
                                data = json.loads(message)
                                message_type = data.get('type')
                                chunk_id = data.get('chunk_id')
                                sequence = data.get('sequence')
                                
                                self.logger.debug("Received message", {
                                    "type": message_type,
                                    "chunk_id": chunk_id,
                                    "sequence": sequence,
                                    "size_bytes": len(message)
                                })
                                
                                if message_type == 'audio-data':
                                    audio_size = len(data.get('audio_data', ''))
                                    self.logger.info("Received audio data", {
                                        "chunk_id": chunk_id,
                                        "sequence": sequence,
                                        "audio_size": audio_size,
                                        "timestamp": data.get('timestamp')
                                    })
                                    
                                    # Process audio in background task to avoid blocking
                                    asyncio.create_task(self.handle_audio_data(data))
                                    
                                elif message_type in ['offer', 'answer', 'ice-candidate']:
                                    self.logger.debug("Received WebRTC signaling message", {
                                        "type": message_type
                                    })
                                else:
                                    self.logger.warning("Unknown message type", {
                                        "type": message_type
                                    })


                            except json.JSONDecodeError as e:
                                self.logger.error("Failed to parse message", {
                                    "error": str(e),
                                    "message_preview": message[:100] + "..." if len(message) > 100 else message
                                })
                            except Exception as e:
                                self.logger.error("Error processing message", {
                                    "error": str(e),
                                    "type": message_type if 'message_type' in locals() else 'unknown'
                                }, exc_info=True)
                            
                    except websockets.exceptions.ConnectionClosed as e:
                        self.logger.warning("Connection closed", {
                            "code": e.code,
                            "reason": e.reason,
                            "clean": e.code == 1000
                        })
                    except Exception as e:
                        self.logger.error("WebSocket error", {
                            "error": str(e)
                        }, exc_info=True)
                    finally:
                        self.websocket = None
                        
            except Exception as e:
                self.logger.error("Connection error", {
                    "error": str(e),
                    "attempt": retry_count + 1
                }, exc_info=True)
                
                retry_count += 1
                if retry_count >= max_retries:
                    self.logger.error("Max retries reached", {
                        "max_retries": max_retries,
                        "total_attempts": retry_count
                    })
                    # Wait longer before trying again
                    await asyncio.sleep(30)
                    retry_count = 0
                    continue
                
                # Exponential backoff with max of 32 seconds
                retry_delay = min(2 ** retry_count, 32)
                self.logger.info("Retrying connection", {
                    "delay_seconds": retry_delay,
                    "next_attempt": retry_count + 1,
                    "max_retries": max_retries
                })
                await asyncio.sleep(retry_delay)

async def main():
    logger = Logger("Main")
    logger.info("Starting AI Server application")
    
    ai_server = AIServer()
    room_id = "test-room"  # You can make this configurable
    
    while True:
        try:
            await ai_server.connect_to_signaling_server(room_id)
        except Exception as e:
            logger.error("Connection error", {"error": str(e)}, exc_info=True)
        
        # Wait before reconnecting
        logger.info("Waiting before reconnection attempt")
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
