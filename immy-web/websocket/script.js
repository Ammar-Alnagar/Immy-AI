let ws;
let mediaRecorder;
let audioContext;
const ttsAudioDiv = document.getElementById('tts-audio');
const startBtn = document.getElementById('startBtn');
const statusDiv = document.getElementById('status');
const chatContainer = document.getElementById('chat-container');
let isListening = false;
let currentLLMResponse = '';
let audioQueue = [];
let isPlaying = false;
let currentAudio = null;
let chunkMetrics = new Map();
let messageQueue = new Map(); // Map of chunk IDs to message divs
let ttsSequences = new Map(); // Map of chunk IDs to next expected TTS sequence

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
    
    // Ensure all metrics are valid numbers, use 0 as fallback
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

function connectWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    log('Connecting to WebSocket...');
    
    ws.onopen = () => {
        log('WebSocket connected');
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        // Handle ping messages silently
        if (data.type === 'ping') {
            return;
        }
        
        log(`Received message: ${JSON.stringify(data)}`);
        
        // Handle streaming LLM chunks
        if (data.type === 'llm_chunk') {
            appendToLLMResponse(data.content, data.chunkId);
            log(`LLM Chunk: ${data.content}`);
            
            // Update metrics for LLM processing
            if (data.metrics && data.chunkId) {
                if (!chunkMetrics.has(data.chunkId)) {
                    chunkMetrics.set(data.chunkId, {});
                }
                const metrics = chunkMetrics.get(data.chunkId);
                Object.assign(metrics, data.metrics);
                updateMetrics(data.chunkId, metrics);
            }
        }
        
        // Handle TTS chunks
        else if (data.type === 'tts_chunk') {
            const audioSrc = 'data:audio/wav;base64,' + data.audio;
            const audio = new Audio(audioSrc);
            
            // Initialize sequence tracking for new chunks
            if (!ttsSequences.has(data.chunkId)) {
                ttsSequences.set(data.chunkId, 1);
                log(`Initializing sequence for chunk ${data.chunkId}`);
            }
            
            audioQueue.push({
                audio: audio,
                text: data.text,
                chunkId: data.chunkId,
                sequence: data.sequence
            });
            
            log(`Added TTS chunk to queue - ChunkID: ${data.chunkId}, Sequence: ${data.sequence}`);
            
            // Update metrics for TTS generation
            if (data.metrics && data.chunkId) {
                if (!chunkMetrics.has(data.chunkId)) {
                    chunkMetrics.set(data.chunkId, {});
                }
                const metrics = chunkMetrics.get(data.chunkId);
                Object.assign(metrics, data.metrics);
                updateMetrics(data.chunkId, metrics);
            }
            
            if (!isPlaying) {
                isPlaying = true;
                playNextInQueue();
            }
        }
        
        // Handle final response
        else if (data.type === 'final_response') {
            if (data.transcription) {
                addMessage(data.transcription, true);
                log(`Transcription: ${data.transcription}`);
            }
            
            // Finalize the streaming response
            const streamingResponse = messageQueue.get(data.chunkId);
            if (streamingResponse) {
                streamingResponse.classList.remove('llm-response-streaming');
            }
        }
        
        // Handle errors
        else if (data.error) {
            statusDiv.textContent = `Error: ${data.error}`;
            log(`Error: ${data.error}`);
            startBtn.classList.remove('recording');
            isListening = false;
        }
    };

    ws.onclose = function(event) {
        log('WebSocket connection closed');
        startBtn.classList.remove('recording');
        isListening = false;
        statusDiv.textContent = "Click the microphone to start";
        
        // Clean up resources
        stopCurrentAudio();
        resetLLMResponse();
        chunkMetrics.clear();
        
        // Attempt to reconnect if not intentionally stopped
        if (isListening) {
            log('Connection lost. Attempting to reconnect...');
            statusDiv.textContent = "Connection lost. Reconnecting...";
            setTimeout(connectWebSocket, 2000);
        }
    };

    ws.onerror = function(error) {
        log(`WebSocket error: ${error}`);
        statusDiv.textContent = `Connection error. Please try again.`;
        startBtn.classList.remove('recording');
        isListening = false;
        
        // Clean up resources on error
        stopCurrentAudio();
        resetLLMResponse();
        chunkMetrics.clear();
    };
}

let audioProcessor = null;
const SILENCE_THRESHOLD = 0.03;
const MIN_SILENCE_LENGTH = 1000; // ms

function handleAudioProcess(inputData, audioChunks, isSpeaking, silenceStart, chunkCounter) {
    const volumeLevel = Math.max(...Array.from(inputData).map(Math.abs));
    const containsSpeech = volumeLevel > SILENCE_THRESHOLD;
    
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
        }
        audioChunks.push(new Float32Array(inputData));
    } else {
        if (isSpeaking) {
            if (!silenceStart) {
                silenceStart = Date.now();
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
                        log('Sending audio chunk ' + chunkCounter);
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({
                                audio_data: base64Audio,
                                mime_type: 'audio/wav',
                                chunk_id: Date.now(),
                                sequence: chunkCounter
                            }));
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
            }
        }
        // Still collect audio during silence to maintain context
        audioChunks.push(new Float32Array(inputData));
    }
    
    return { isSpeaking, silenceStart, chunkCounter };
}

async function toggleRecording() {
    if (isListening) {
        isListening = false;  // Set this first to prevent reconnection attempts
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
        audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        
        let audioChunks = [];
        let isSpeaking = false;
        let silenceStart = null;
        let chunkCounter = 0;
        
        audioProcessor.onaudioprocess = function(e) {
            const inputData = e.inputBuffer.getChannelData(0);
            const result = handleAudioProcess(inputData, audioChunks, isSpeaking, silenceStart, chunkCounter);
            isSpeaking = result.isSpeaking;
            silenceStart = result.silenceStart;
            chunkCounter = result.chunkCounter;
        };
        
        source.connect(audioProcessor);
        audioProcessor.connect(audioContext.destination);
        
        log('Started recording');
        startBtn.classList.add('recording');
        isListening = true;
        statusDiv.textContent = "Listening...";
        
        // Reset all state
        resetLLMResponse();
        stopCurrentAudio();
        chunkMetrics.clear();
        
        connectWebSocket();
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
        if (audioProcessor) {
            audioProcessor.disconnect();
            audioProcessor = null;
        }
        audioContext.close();
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
        log('Sending stop signal to server');
        ws.send(JSON.stringify({ stop: true }));
        ws.close();
    }
    startBtn.classList.remove('recording');
    isListening = false;
    statusDiv.textContent = "Click the microphone to start";
    
    // Clear all state
    stopCurrentAudio();
    resetLLMResponse();
    
    log('Recording stopped');
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
