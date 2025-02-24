let peerConnection;
let signalingSocket;
let mediaRecorder;
let audioContext;
let isListening = false;
let currentLLMResponse = '';
let audioQueue = [];
let isPlaying = false;
let currentAudio = null;
let chunkMetrics = new Map();
let messageQueue = new Map();
let ttsSequences = new Map();
let dataChannel;
let roomId;

const ttsAudioDiv = document.getElementById('tts-audio');
const startBtn = document.getElementById('startBtn');
const statusDiv = document.getElementById('status');
const chatContainer = document.getElementById('chat-container');
const connectionStatus = document.getElementById('connectionStatus');

const ICE_SERVERS = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
    ]
};

function updateConnectionStatus(status, isConnected = false) {
    connectionStatus.textContent = status;
    connectionStatus.className = 'connection-status ' + 
        (isConnected ? 'connected' : 'disconnected');
}

function addMessage(text, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
    messageDiv.textContent = text;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return messageDiv;
}

function appendToLLMResponse(chunk, chunkId) {
    let llmMessageDiv = messageQueue.get(chunkId);
    if (!llmMessageDiv) {
        llmMessageDiv = document.createElement('div');
        llmMessageDiv.className = 'message assistant-message llm-response-streaming';
        llmMessageDiv.dataset.chunkId = chunkId;
        llmMessageDiv.textContent = '';
        chatContainer.appendChild(llmMessageDiv);
        messageQueue.set(chunkId, llmMessageDiv);
    }
    
    llmMessageDiv.textContent += chunk;
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function resetLLMResponse() {
    messageQueue.forEach((div) => div.remove());
    messageQueue.clear();
    ttsSequences.clear();
    currentLLMResponse = '';
}

function log(message) {
    console.log(`${new Date().toISOString()}: ${message}`);
}

function handleResponse(data) {
    if (data.text) {
        appendToLLMResponse(data.text, data.chunkId);
    }
    
    if (data.audio) {
        const audioSrc = 'data:audio/wav;base64,' + data.audio;
        const audio = new Audio(audioSrc);
        
        // Log audio data received
        log(`Received audio data - ChunkID: ${data.chunkId}, Sequence: ${data.sequence}, Size: ${data.audio.length} bytes`);
        
        // Ensure audio is loaded before adding to queue
        audio.addEventListener('loadeddata', () => {
            log(`Audio loaded - ChunkID: ${data.chunkId}, Sequence: ${data.sequence}`);
            
            if (!ttsSequences.has(data.chunkId)) {
                ttsSequences.set(data.chunkId, 1);
                log(`Initialized TTS sequence for chunk ${data.chunkId}`);
            }
            
            audioQueue.push({
                audio: audio,
                text: data.text,
                chunkId: data.chunkId,
                sequence: data.sequence
            });
            log(`Added audio to queue - Queue size: ${audioQueue.length}`);
            
            if (!isPlaying) {
                log('Starting audio playback');
                isPlaying = true;
                playNextInQueue();
            } else {
                log('Audio playback already in progress');
            }
        });

        // Handle audio loading errors
        audio.addEventListener('error', (e) => {
            log(`Error loading audio - ChunkID: ${data.chunkId}, Sequence: ${data.sequence}, Error: ${e.error}`);
        });

        // Start loading the audio
        audio.load();
    }
    
    if (data.metrics) {
        if (!chunkMetrics.has(data.chunkId)) {
            chunkMetrics.set(data.chunkId, {});
        }
        const metrics = chunkMetrics.get(data.chunkId);
        Object.assign(metrics, data.metrics);
        updateMetrics(data.chunkId, metrics);
    }
}

function playNextInQueue() {
    if (audioQueue.length === 0 || !isPlaying) {
        isPlaying = false;
        currentAudio = null;
        log('Audio queue empty or playback stopped');
        return;
    }

    // Sort queue by chunk ID and sequence
    audioQueue.sort((a, b) => {
        if (a.chunkId === b.chunkId) {
            return a.sequence - b.sequence;
        }
        return a.chunkId - b.chunkId;
    });

    const nextItem = audioQueue[0];
    log(`Next audio item - ChunkID: ${nextItem.chunkId}, Sequence: ${nextItem.sequence}`);
    
    // Check if this is the next expected sequence for this chunk
    const expectedSequence = ttsSequences.get(nextItem.chunkId) || 1;
    log(`Expected sequence for chunk ${nextItem.chunkId}: ${expectedSequence}`);
    
    if (nextItem.sequence !== expectedSequence) {
        log(`Sequence mismatch - Expected: ${expectedSequence}, Got: ${nextItem.sequence}`);
        // Look for the correct sequence in the queue
        const correctItem = audioQueue.find(item => 
            item.chunkId === nextItem.chunkId && item.sequence === expectedSequence
        );
        if (correctItem) {
            // Move the correct item to the front
            audioQueue.splice(audioQueue.indexOf(correctItem), 1);
            audioQueue.unshift(correctItem);
            log(`Found correct sequence item, moved to front of queue`);
        } else {
            // Remove the out-of-sequence item
            audioQueue.shift();
            log(`No matching sequence found, removing item from queue`);
            playNextInQueue();
            return;
        }
    }

    currentAudio = nextItem.audio;

    // Log metrics for this chunk's TTS playback start
    if (chunkMetrics.has(nextItem.chunkId)) {
        const metrics = chunkMetrics.get(nextItem.chunkId);
        metrics.ttsPlaybackStart = Date.now();
        updateMetrics(nextItem.chunkId, metrics);
    }

    currentAudio.onended = () => {
        // Log metrics for this chunk's TTS playback end
        if (chunkMetrics.has(nextItem.chunkId)) {
            const metrics = chunkMetrics.get(nextItem.chunkId);
            metrics.ttsPlaybackEnd = Date.now();
            updateMetrics(nextItem.chunkId, metrics);
        }

        // Update expected sequence for this chunk
        ttsSequences.set(nextItem.chunkId, expectedSequence + 1);

        audioQueue.shift(); // Remove the played item
        currentAudio = null;
        if (isPlaying) {
            playNextInQueue(); // Play the next item if any and if we should still be playing
        }
    };

    currentAudio.play().catch(error => {
        log(`Error playing audio: ${error}`);
        audioQueue.shift(); // Remove the problematic item
        currentAudio = null;
        if (isPlaying) {
            playNextInQueue(); // Try the next item
        }
    });
}

function stopCurrentAudio() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }
    audioQueue = [];
    isPlaying = false;
}

function updateMetrics(chunkId, metrics) {
    const metricsContent = document.getElementById('metricsContent');
    const existingMetric = document.getElementById(`metric-${chunkId}`);
    
    const processingTime = (metrics.processingEnd && metrics.processingStart) ? 
        Math.max(0, metrics.processingEnd - metrics.processingStart) : 0;
    const transcriptionTime = (metrics.transcriptionEnd && metrics.transcriptionStart) ? 
        Math.max(0, metrics.transcriptionEnd - metrics.transcriptionStart) : 0;
    const llmTime = (metrics.llmEnd && metrics.llmStart) ? 
        Math.max(0, metrics.llmEnd - metrics.llmStart) : 0;
    const ttsTime = (metrics.ttsEnd && metrics.ttsStart) ? 
        Math.max(0, metrics.ttsEnd - metrics.ttsStart) : 0;
    const playbackTime = (metrics.ttsPlaybackEnd && metrics.ttsPlaybackStart) ? 
        Math.max(0, metrics.ttsPlaybackEnd - metrics.ttsPlaybackStart) : null;

    const metricHTML = `
        <div class="metric-section">
            <span class="metric-label">Chunk ${chunkId}:</span>
        </div>
        <div class="metric-section">
            <span class="metric-label">Audio Processing:</span>
            <span class="metric-time">${processingTime}ms</span>
        </div>
        <div class="metric-section">
            <span class="metric-label">Transcription:</span>
            <span class="metric-time">${transcriptionTime}ms</span>
        </div>
        <div class="metric-section">
            <span class="metric-label">LLM Response:</span>
            <span class="metric-time">${llmTime}ms</span>
        </div>
        <div class="metric-section">
            <span class="metric-label">TTS Generation:</span>
            <span class="metric-time">${ttsTime}ms</span>
        </div>
        ${playbackTime ? `
        <div class="metric-section">
            <span class="metric-label">TTS Playback:</span>
            <span class="metric-time">${playbackTime}ms</span>
        </div>
        ` : ''}
        <div class="metric-section">
            <span class="metric-label">Total Processing:</span>
            <span class="metric-time">${metrics.totalTime}ms</span>
        </div>
    `;

    if (existingMetric) {
        existingMetric.innerHTML = metricHTML;
    } else {
        const metricItem = document.createElement('div');
        metricItem.id = `metric-${chunkId}`;
        metricItem.className = 'metric-item';
        metricItem.innerHTML = metricHTML;
        metricsContent.appendChild(metricItem);
        metricsContent.scrollTop = metricsContent.scrollHeight;
    }
}

async function initWebRTC() {
    const maxRetries = 3;
    let retryCount = 0;
    let retryDelay = 1000; // Start with 1 second delay

    async function attemptConnection() {
        try {
            // Use fixed room ID to match server
            roomId = "test-room";
            log('Attempting connection to room: ' + roomId);
            
            // Connect to signaling server as client
            signalingSocket = new WebSocket(`ws://${window.location.host}/ws/client/${roomId}`);
            
            signalingSocket.onopen = () => {
                log('Connected to signaling server in room: ' + roomId);
                updateConnectionStatus('Connected to signaling server in room: ' + roomId, true);
                setupPeerConnection();
                retryCount = 0; // Reset retry count on successful connection
            };
        
            signalingSocket.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'offer' || data.type === 'answer' || data.type === 'ice-candidate') {
                    // Handle WebRTC signaling messages
                    if (data.type === 'offer') await handleOffer(data);
                    else if (data.type === 'answer') await handleAnswer(data);
                    else if (data.type === 'ice-candidate') await handleIceCandidate(data);
                } else if (data.type === 'response') {
                    // Handle audio/text responses
                    handleResponse(data);
                } else if (data.type === 'final') {
                    handleFinalResponse(data);
                } else if (data.type === 'error') {
                    log(`Error from server: ${data.message}`);
                    updateConnectionStatus(data.message);
                }
            };
            
            signalingSocket.onclose = async () => {
                log('Disconnected from signaling server');
                updateConnectionStatus('Disconnected from signaling server');
                cleanup();

                // Attempt to reconnect if not manually stopped
                if (isListening && retryCount < maxRetries) {
                    retryCount++;
                    log(`Attempting reconnection ${retryCount}/${maxRetries} in ${retryDelay/1000}s`);
                    updateConnectionStatus(`Reconnecting... Attempt ${retryCount}/${maxRetries}`);
                    await new Promise(resolve => setTimeout(resolve, retryDelay));
                    retryDelay *= 2; // Exponential backoff
                    await attemptConnection();
                } else if (retryCount >= maxRetries) {
                    log('Max reconnection attempts reached');
                    updateConnectionStatus('Connection failed after max retries');
                    stopRecording();
                }
            };
            
            signalingSocket.onerror = (error) => {
                log(`WebSocket error: ${error}`);
                updateConnectionStatus('Connection error');
                // Let onclose handle reconnection
            };
            
        } catch (error) {
            log(`Error initializing connection: ${error}`);
            updateConnectionStatus('Failed to initialize connection');
            
            // Attempt to reconnect
            if (retryCount < maxRetries) {
                retryCount++;
                log(`Attempting reconnection ${retryCount}/${maxRetries} in ${retryDelay/1000}s`);
                updateConnectionStatus(`Reconnecting... Attempt ${retryCount}/${maxRetries}`);
                await new Promise(resolve => setTimeout(resolve, retryDelay));
                retryDelay *= 2; // Exponential backoff
                await attemptConnection();
            } else {
                log('Max reconnection attempts reached');
                updateConnectionStatus('Connection failed after max retries');
                stopRecording();
            }
        }
    }

    // Start the connection attempt
    await attemptConnection();
}

async function setupPeerConnection() {
    try {
        peerConnection = new RTCPeerConnection(ICE_SERVERS);
        
        // Create data channel
        dataChannel = peerConnection.createDataChannel('audioData');
        setupDataChannel();
        
        // Handle ICE candidates
        peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                signalingSocket.send(JSON.stringify({
                    type: 'ice-candidate',
                    candidate: event.candidate
                }));
            }
        };
        
        // Create and send offer
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        signalingSocket.send(JSON.stringify({
            type: 'offer',
            offer: offer
        }));
        
    } catch (error) {
        log(`Error setting up peer connection: ${error}`);
        updateConnectionStatus('Failed to setup peer connection');
    }
}

function setupDataChannel() {
    dataChannel.onopen = () => {
        log('Data channel opened in room: ' + roomId);
        updateConnectionStatus('WebRTC connection established in room: ' + roomId, true);
    };
    
    dataChannel.onclose = () => {
        log('Data channel closed');
        updateConnectionStatus('WebRTC connection closed');
    };
    
    dataChannel.onerror = (error) => {
        log(`Data channel error: ${error}`);
        updateConnectionStatus('WebRTC connection error');
    };
    
    dataChannel.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'response') {
            handleResponse(data);
        } else if (data.type === 'final') {
            handleFinalResponse(data);
        }
    };
}

async function createAndSendOffer() {
    try {
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        
        signalingSocket.send(JSON.stringify({
            type: 'offer',
            offer: offer
        }));
        
    } catch (error) {
        log(`Error creating offer: ${error}`);
        updateConnectionStatus('Failed to create connection offer');
    }
}

async function handleOffer(data) {
    try {
        await peerConnection.setRemoteDescription(new RTCSessionDescription(data.offer));
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);
        
        signalingSocket.send(JSON.stringify({
            type: 'answer',
            answer: answer
        }));
        
    } catch (error) {
        log(`Error handling offer: ${error}`);
    }
}

async function handleAnswer(data) {
    try {
        await peerConnection.setRemoteDescription(new RTCSessionDescription(data.answer));
    } catch (error) {
        log(`Error handling answer: ${error}`);
    }
}

async function handleIceCandidate(data) {
    try {
        if (data.candidate) {
            await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
        }
    } catch (error) {
        log(`Error handling ICE candidate: ${error}`);
    }
}

function handleFinalResponse(data) {
    if (data.transcription) {
        addMessage(data.transcription, true);
    }
    
    const streamingResponse = messageQueue.get(data.chunkId);
    if (streamingResponse) {
        streamingResponse.classList.remove('llm-response-streaming');
    }
    
    if (data.metrics) {
        updateMetrics(data.chunkId, data.metrics);
    }
}

const SILENCE_THRESHOLD = 0.03;
const MIN_SILENCE_LENGTH = 1000; // ms

function handleAudioProcess(inputData, audioChunks, isSpeaking, silenceStart, chunkCounter) {
    const volumeLevel = Math.max(...Array.from(inputData).map(Math.abs));
    const containsSpeech = volumeLevel > SILENCE_THRESHOLD;
    
    // Log audio levels for debugging
    if (containsSpeech) {
        log(`Speech detected - Volume: ${volumeLevel.toFixed(3)}`);
    }
    
    // Check if we need to stop TTS playback
    if (containsSpeech && (audioQueue.length > 0 || currentAudio)) {
        log('Speech detected during TTS playback, stopping audio');
        stopCurrentAudio();
        if (isListening) {
            statusDiv.textContent = "Listening...";
        }
    }
    
    if (containsSpeech) {
        if (!isSpeaking) {
            log('Speech detected, starting new chunk');
            isSpeaking = true;
            silenceStart = null;
            statusDiv.textContent = "Speech detected...";
        }
        audioChunks.push(new Float32Array(inputData));
    } else {
        if (isSpeaking) {
            if (!silenceStart) {
                silenceStart = Date.now();
                statusDiv.textContent = "Processing...";
            } else if (Date.now() - silenceStart >= MIN_SILENCE_LENGTH) {
                // Silence detected for long enough - send the chunk
                isSpeaking = false;
                
                if (audioChunks.length > 0) {
                    // Concatenate chunks
                    const concatenated = new Float32Array(audioChunks.reduce((acc, chunk) => acc + chunk.length, 0));
                    let offset = 0;
                    audioChunks.forEach(chunk => {
                        concatenated.set(chunk, offset);
                        offset += chunk.length;
                    });
                    
                    // Convert to WAV
                    const wavBuffer = float32ToWav(concatenated, 16000);
                    const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
                    
                    // Send to server
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const base64Audio = reader.result.split(',')[1];
                        const timestamp = Date.now();
                        const chunk_id = timestamp;
                        
                        log(`Preparing audio chunk ${chunkCounter} | Size: ${base64Audio.length} bytes`);
                        
                        if (signalingSocket && signalingSocket.readyState === WebSocket.OPEN) {
                            const message = {
                                type: 'audio-data',
                                audio_data: base64Audio,
                                chunk_id: chunk_id,
                                sequence: chunkCounter,
                                timestamp: timestamp
                            };
                            
                            signalingSocket.send(JSON.stringify(message));
                            log(`Sent audio chunk ${chunkCounter} to room: ${roomId}`);
                        } else {
                            log('WebSocket not connected, cannot send audio');
                            stopRecording();
                        }
                    };
                    reader.readAsDataURL(wavBlob);
                    
                    // Reset for next chunk
                    audioChunks.length = 0;
                    chunkCounter++;
                }
                
                statusDiv.textContent = "Listening...";
            }
        }
        // Still collect audio during silence to maintain context
        audioChunks.push(new Float32Array(inputData));
    }
    
    return { isSpeaking, silenceStart, chunkCounter };
}

async function toggleRecording() {
    if (isListening) {
        stopRecording();
        return;
    }
    
    try {
        log('Requesting microphone access...');
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000
            } 
        });
        log('Microphone access granted');
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });
        
        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        let audioChunks = [];
        let isSpeaking = false;
        let silenceStart = null;
        let chunkCounter = 0;
        
        processor.onaudioprocess = function(e) {
            const inputData = e.inputBuffer.getChannelData(0);
            const result = handleAudioProcess(inputData, audioChunks, isSpeaking, silenceStart, chunkCounter);
            isSpeaking = result.isSpeaking;
            silenceStart = result.silenceStart;
            chunkCounter = result.chunkCounter;
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        log('Started recording');
        startBtn.classList.add('recording');
        isListening = true;
        statusDiv.textContent = "Listening...";
        
        resetLLMResponse();
        stopCurrentAudio();
        chunkMetrics.clear();
        
        // Initialize WebRTC connection
        await initWebRTC();
        
    } catch (err) {
        log(`Error: ${err.message}`);
        console.error("Error accessing microphone:", err);
        statusDiv.textContent = `Error: ${err.message}`;
        startBtn.classList.remove('recording');
        isListening = false;
    }
}

function stopRecording() {
    log('Stopping recording...');
    
    if (audioContext) {
        audioContext.close();
    }
    
    cleanup();
    
    startBtn.classList.remove('recording');
    isListening = false;
    statusDiv.textContent = "Click the microphone to start";
    
    stopCurrentAudio();
    resetLLMResponse();
    
    log('Recording stopped');
}

function cleanup() {
    if (dataChannel) {
        dataChannel.close();
    }
    
    if (peerConnection) {
        peerConnection.close();
    }
    
    if (signalingSocket && signalingSocket.readyState === WebSocket.OPEN) {
        signalingSocket.close();
    }
}

// Utility function to convert Float32Array to WAV buffer
function float32ToWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);
    return buffer;
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function floatTo16BitPCM(view, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, input[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

// Initialize click handler after DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    startBtn.onclick = toggleRecording;
});
