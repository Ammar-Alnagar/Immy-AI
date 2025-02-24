from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
from typing import Dict, Set, Optional
import asyncio
import os
from pathlib import Path
from datetime import datetime
from services.logger import Logger

app = FastAPI()

# Get the directory containing this file
BASE_DIR = Path(__file__).resolve().parent

# Mount the static files directory
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

class Connection:
    def __init__(self, websocket: WebSocket, role: str):
        self.websocket = websocket
        self.role = role  # 'client' or 'server'
        self.connected_at = datetime.now().timestamp()

class Room:
    def __init__(self):
        self.client: Optional[Connection] = None
        self.server: Optional[Connection] = None
        self.created_at = datetime.now().timestamp()

class SignalingServer:
    def __init__(self):
        self.logger = Logger("SignalingServer")
        self.rooms: Dict[str, Room] = {}
        self.logger.info("SignalingServer initialized")
    
    def get_or_create_room(self, room_id: str) -> Room:
        if room_id not in self.rooms:
            self.rooms[room_id] = Room()
            self.logger.info("Created new room", {"room_id": room_id})
        return self.rooms[room_id]
    
    async def connect(self, websocket: WebSocket, room_id: str, role: str):
        connection = None
        try:
            # Log connection attempt before accepting
            self.logger.info("New connection attempt", {
                "room_id": room_id,
                "role": role,
                "remote": websocket.client.host
            })
            
            await websocket.accept()
            room = self.get_or_create_room(room_id)
            connection = Connection(websocket, role)
            
            # Log room state
            self.logger.debug("Room state", {
                "room_id": room_id,
                "client_exists": bool(room.client),
                "server_exists": bool(room.server),
                "room_age": round(datetime.now().timestamp() - room.created_at, 2)
            })
            
            # Assign connection based on role
            if role == 'client':
                if room.client:
                    self.logger.warning("Room already has a client", {"room_id": room_id})
                    await websocket.send_json({"error": "Room already has a client"})
                    return
                room.client = connection
                self.logger.info("Client connected", {"room_id": room_id})
            else:  # role == 'server'
                if room.server:
                    self.logger.warning("Room already has a server", {"room_id": room_id})
                    await websocket.send_json({"error": "Room already has a server"})
                    return
                room.server = connection
                self.logger.info("Server connected", {"room_id": room_id})

            # Main message handling loop
            while True:
                try:
                    message = await websocket.receive_json()
                    message_type = message.get('type')
                    message_size = len(str(message))
                    
                    self.logger.debug("Received message", {
                        "room_id": room_id,
                        "role": role,
                        "type": message_type,
                        "size_bytes": message_size,
                        "chunk_id": message.get('chunk_id'),
                        "sequence": message.get('sequence')
                    })
                    
                    # Monitor large messages
                    if message_size > 1000000:  # 1MB
                        self.logger.warning("Large message received", {
                            "size_mb": round(message_size / 1000000, 2),
                            "type": message_type
                        })
                
                    if message_type == 'audio-data':
                        # Forward audio data to server
                        if role == 'client' and room.server:
                            audio_size = len(message.get('audio_data', ''))
                            self.logger.info("Received audio data from client", {
                                "room_id": room_id,
                                "chunk_id": message.get('chunk_id'),
                                "sequence": message.get('sequence'),
                                "audio_size": audio_size,
                                "server_connected": bool(room.server)
                            })
                            
                            # Check audio size
                            if audio_size == 0:
                                self.logger.warning("Empty audio data received", {
                                    "chunk_id": message.get('chunk_id')
                                })
                                continue
                                
                            await self.forward_message(room, connection, message)
                        else:
                            self.logger.warning("Cannot forward audio data", {
                                "room_id": room_id,
                                "has_server": bool(room.server),
                                "role": role
                            })
                    elif message_type in ['offer', 'answer', 'ice-candidate']:
                        # Forward WebRTC signaling messages
                        self.logger.debug("Received WebRTC signaling message", {
                            "room_id": room_id,
                            "type": message_type,
                            "role": role,
                            "target_role": "server" if role == "client" else "client"
                        })
                        await self.forward_message(room, connection, message)
                    elif message_type in ['response', 'final']:
                        # Forward TTS responses and final results
                        self.logger.debug("Forwarding response", {
                            "room_id": room_id,
                            "type": message_type,
                            "chunk_id": message.get('chunkId') or message.get('chunk_id'),
                            "sequence": message.get('sequence')
                        })
                        await self.forward_message(room, connection, message)
                    elif message_type == 'leave':
                        self.logger.info("Client requested to leave", {
                            "room_id": room_id,
                            "role": role,
                            "duration": round(datetime.now().timestamp() - connection.connected_at, 2)
                        })
                        break
                    else:
                        self.logger.warning("Unknown message type", {
                            "room_id": room_id,
                            "type": message_type,
                            "role": role
                        })
                        
                except json.JSONDecodeError as e:
                    self.logger.error("Invalid JSON message", {
                        "error": str(e),
                        "preview": message[:100] if isinstance(message, str) else str(message)[:100]
                    })
                
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.warning("WebSocket connection closed", {
                "room_id": room_id,
                "role": role,
                "code": e.code,
                "reason": e.reason,
                "clean": e.code == 1000,
                "duration": round(datetime.now().timestamp() - connection.connected_at, 2) if connection else 0
            })
        except Exception as e:
            self.logger.error("WebSocket connection error", {
                "room_id": room_id,
                "role": role,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration": round(datetime.now().timestamp() - connection.connected_at, 2) if connection else 0
            }, exc_info=True)
        finally:
            if connection:
                # Calculate connection duration
                duration = round(datetime.now().timestamp() - connection.connected_at, 2)
                
                # Clean up connection
                if role == 'client':
                    if room.client == connection:
                        room.client = None
                        self.logger.info("Client disconnected", {
                            "room_id": room_id,
                            "duration_sec": duration,
                            "clean_disconnect": True
                        })
                else:
                    if room.server == connection:
                        room.server = None
                        self.logger.info("Server disconnected", {
                            "room_id": room_id,
                            "duration_sec": duration,
                            "clean_disconnect": True
                        })
                
                # Remove room if empty
                if not room.client and not room.server:
                    room_duration = round(datetime.now().timestamp() - room.created_at, 2)
                    del self.rooms[room_id]
                    self.logger.info("Room closed", {
                        "room_id": room_id,
                        "duration_sec": room_duration,
                        "total_connections": 2 if role == 'client' else 1
                    })
                else:
                    self.logger.debug("Room state after disconnect", {
                        "room_id": room_id,
                        "has_client": bool(room.client),
                        "has_server": bool(room.server)
                    })

    async def forward_message(self, room: Room, sender: Connection, message: dict):
        """Forward message to the other peer in the room."""
        try:
            message_type = message.get('type', 'unknown')
            chunk_id = message.get('chunkId') or message.get('chunk_id')
            
            self.logger.debug("Forwarding message", {
                "from_role": sender.role,
                "type": message_type,
                "chunk_id": chunk_id,
                "size_bytes": len(str(message))
            })
            
            # Determine target connection
            target_connection = room.server if sender.role == 'client' else room.client
            
            if not target_connection:
                self.logger.warning("No target connection available", {
                    "from_role": sender.role,
                    "type": message_type,
                    "chunk_id": chunk_id
                })
                return
            
            # Forward the message
            start_time = datetime.now().timestamp()
            await target_connection.websocket.send_json(message)
            duration = round((datetime.now().timestamp() - start_time) * 1000, 2)
            
            # Log success based on message type
            if message_type == 'audio-data':
                self.logger.info("Audio data forwarded", {
                    "from": sender.role,
                    "to": target_connection.role,
                    "chunk_id": chunk_id,
                    "duration_ms": duration,
                    "audio_size": len(message.get('audio_data', ''))
                })
            else:
                self.logger.info("Message forwarded", {
                    "from": sender.role,
                    "to": target_connection.role,
                    "type": message_type,
                    "chunk_id": chunk_id,
                    "duration_ms": duration
                })
            
        except Exception as e:
            self.logger.error("Failed to forward message", {
                "error": str(e),
                "type": message_type,
                "from_role": sender.role,
                "chunk_id": chunk_id
            })

signaling_server = SignalingServer()

@app.get("/")
async def get():
    """Serve the main HTML page."""
    return FileResponse(str(BASE_DIR / 'index.html'))

@app.websocket("/ws/{role}/{room_id}")
async def websocket_endpoint(websocket: WebSocket, role: str, room_id: str):
    """Handle WebSocket connections for signaling."""
    if role not in ['client', 'server']:
        await websocket.close(code=1008, reason="Invalid role")
        return
    await signaling_server.connect(websocket, room_id, role)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
