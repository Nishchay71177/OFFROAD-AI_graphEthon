// ================ WEBSOCKET CONNECTION ================

const socket = io();
let isConnected = false;
let isStreamingActive = false;

socket.on('connect', () => {
    isConnected = true;
    updateConnectionStatus('🟢 Connected', 'connected');
    console.log('✅ Connected to server');
});

socket.on('disconnect', () => {
    isConnected = false;
    updateConnectionStatus('🔴 Disconnected', 'disconnected');
    console.log('❌ Disconnected from server');
});

socket.on('frame_result', (data) => {
    const result = data.result;
    const history = data.history;
    const userStats = data.user_stats;
    
    if (!result.error) {
        displayLiveResult(result);
        updateHistory(history);
        updateFrameCounter(userStats.frames_processed);
    } else {
        console.error('Error:', result.error);
    }
});

socket.on('error', (data) => {
    console.error('WebSocket error:', data.message);
    alert('Error: ' + data.message);
});

socket.on('history_clear', () => {
    console.log('✅ History cleared');
    updateHistoryUI([]);
});

// ================ CONNECTION STATUS ================

function updateConnectionStatus(text, status) {
    const statusEl = document.getElementById('connectionStatus');
    statusEl.textContent = text;
    statusEl.className = 'connection-status ' + status;
}

// ================ TAB SWITCHING ================

function switchTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    // Remove active class from buttons
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    // Show selected tab
    const selectedTab = document.getElementById(tabName + '-tab');
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // Mark button as active
    event.target?.classList.add('active');
}

// ================ IMAGE UPLOAD & ANALYSIS ================

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
let selectedImageFile = null;

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    selectedImageFile = e.target.files[0];
    if (selectedImageFile) {
        previewImage(selectedImageFile);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.background = 'rgba(56,189,248,0.15)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.background = '';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.background = '';
    selectedImageFile = e.dataTransfer.files[0];
    if (selectedImageFile) {
        previewImage(selectedImageFile);
    }
});

function previewImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('preview').src = e.target.result;
    };
    reader.readAsDataURL(file);
}

async function predictImage() {
    if (!selectedImageFile) {
        alert('Please select an image first');
        return;
    }
    
    const statusEl = document.getElementById('imageStatus');
    statusEl.textContent = '⏳ Analyzing...';
    statusEl.className = 'status analyzing';
    
    try {
        const formData = new FormData();
        formData.append('file', selectedImageFile);
        
        const response = await fetch('/predict-image', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Analysis failed');
        }
        
        const data = await response.json();
        
        // Update UI
        document.getElementById('imageTerrain').textContent = data.terrain;
        document.getElementById('imageConfidence').textContent = data.confidence;
        document.getElementById('imageDecision').textContent = data.decision;
        document.getElementById('imageDescription').textContent = data.description;
        
        if (data.mask) {
            document.getElementById('maskPreview').src = 'data:image/jpeg;base64,' + data.mask;
        }
        
        statusEl.textContent = '✅ Analysis Complete';
        statusEl.className = 'status success';
        
    } catch (error) {
        console.error('Error:', error);
        statusEl.textContent = '❌ Error: ' + error.message;
        statusEl.className = 'status error';
    }
}

// ================ VIDEO UPLOAD & ANALYSIS ================

const dropZoneVideo = document.getElementById('dropZoneVideo');
const fileInputVideo = document.getElementById('fileInputVideo');
let selectedVideoFile = null;

dropZoneVideo.addEventListener('click', () => fileInputVideo.click());

fileInputVideo.addEventListener('change', (e) => {
    selectedVideoFile = e.target.files[0];
    if (selectedVideoFile) {
        console.log('✅ Video selected:', selectedVideoFile.name);
    }
});

dropZoneVideo.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZoneVideo.style.background = 'rgba(56,189,248,0.15)';
});

dropZoneVideo.addEventListener('dragleave', () => {
    dropZoneVideo.style.background = '';
});

dropZoneVideo.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZoneVideo.style.background = '';
    selectedVideoFile = e.dataTransfer.files[0];
    if (selectedVideoFile) {
        console.log('✅ Video selected:', selectedVideoFile.name);
    }
});

async function predictVideo() {
    if (!selectedVideoFile) {
        alert('Please select a video first');
        return;
    }
    
    const statusEl = document.getElementById('videoStatus');
    statusEl.textContent = '⏳ Processing video (extracting 7 frames)...';
    statusEl.className = 'status analyzing';
    
    try {
        const formData = new FormData();
        formData.append('file', selectedVideoFile);
        
        const response = await fetch('/predict-video', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Video analysis failed');
        }
        
        const data = await response.json();
        
        // Display results
        document.getElementById('videoResultsContainer').style.display = 'block';
        document.getElementById('videoFinalTerrain').textContent = data.final_terrain || '-';
        document.getElementById('videoFinalDecision').textContent = data.final_decision || '-';
        document.getElementById('videoFinalDescription').textContent = data.final_decision_description || '-';
        
        // Display frame results
        const framesContainer = document.getElementById('videoFramesResults');
        framesContainer.innerHTML = data.frame_predictions.map((frame, idx) => `
            <div class="glass history-item">
                <div class="history-header">
                    <h4>Frame ${frame.part} of 7</h4>
                </div>
                
                <div class="metric">
                    <span class="label">Terrain:</span>
                    <span class="value terrain-${frame.terrain.toLowerCase().replace(/\s+/g, '-')}">${frame.terrain}</span>
                </div>
                
                <div class="metric">
                    <span class="label">Confidence:</span>
                    <span class="value">${frame.confidence}</span>
                </div>
                
                <div class="metric">
                    <span class="label">Decision:</span>
                    <span class="value decision-${frame.decision.replace(/\s+/g, '-').toLowerCase()}">${frame.decision}</span>
                </div>
                
                <p class="description">${frame.decision_description}</p>
                
                ${frame.mask ? `<img src="data:image/jpeg;base64,${frame.mask}" alt="Mask" style="width: 100%; border-radius: 5px; margin-top: 10px;">` : ''}
            </div>
        `).join('');
        
        statusEl.textContent = '✅ Video Analysis Complete - 7 frames processed';
        statusEl.className = 'status success';
        
    } catch (error) {
        console.error('Error:', error);
        statusEl.textContent = '❌ Error: ' + error.message;
        statusEl.className = 'status error';
    }
}

// ================ LIVE STREAM HANDLING ================

let mediaStream = null;
let video = document.getElementById('liveVideo');
let canvas = document.getElementById('captureCanvas');
let ctx = canvas.getContext('2d');
let frameCount = 0;
let lastFrameTime = Date.now();
let fps = 0;

async function startLiveStream() {
    if (!isConnected) {
        alert('Not connected to server. Please wait...');
        return;
    }
    
    try {
        // Request camera access
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'environment'
            }
        });
        
        video.srcObject = mediaStream;
        video.play();
        
        // Enable/disable buttons
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        
        isStreamingActive = true;
        console.log('✅ Camera started');
        
        // Start capturing frames
        captureAndProcessFrame();
        
    } catch (error) {
        console.error('Camera error:', error);
        alert('❌ Cannot access camera: ' + error.message);
    }
}

function stopLiveStream() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    
    video.srcObject = null;
    isStreamingActive = false;
    
    // Reset buttons
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    
    console.log('✅ Camera stopped');
}

function captureAndProcessFrame() {
    if (!isStreamingActive) return;
    
    try {
        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to JPEG base64 (lightweight)
        const frameBase64 = canvas.toDataURL('image/jpeg', 0.7); // 70% quality for compression
        
        // Send to server via WebSocket
        socket.emit('process_frame', {
            frame: frameBase64
        });
        
        frameCount++;
        updateFPS();
        
        // Capture next frame after short delay
        setTimeout(captureAndProcessFrame, 300); // ~3 FPS for consistency
        
    } catch (error) {
        console.error('Frame capture error:', error);
    }
}

function updateFPS() {
    const now = Date.now();
    if (now - lastFrameTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastFrameTime = now;
        document.getElementById('fpsCounter').textContent = fps;
    }
}

// ================ LIVE RESULT DISPLAY ================

function displayLiveResult(result) {
    if (result.error) {
        console.error('Processing error:', result.error);
        return;
    }
    
    const terrain = result.terrain || '-';
    const confidence = result.confidence || '-';
    const decision = result.decision || '-';
    const description = result.description || '-';
    
    // Update current result display
    document.getElementById('liveTerrain').textContent = terrain;
    document.getElementById('liveTerrain').className = 'value terrain-badge terrain-' + terrain.toLowerCase().replace(/\s+/g, '-');
    
    document.getElementById('liveConfidence').textContent = confidence;
    
    document.getElementById('liveDecision').textContent = decision;
    document.getElementById('liveDecision').className = 'value decision-badge decision-' + decision.replace(/\s+/g, '-').toLowerCase();
    
    document.getElementById('liveDescription').textContent = description;
    
    // Display segmentation mask if available
    if (result.mask) {
        document.getElementById('liveMaskPreview').src = 'data:image/jpeg;base64,' + result.mask;
    }
}

// ================ FRAME COUNTER ================

function updateFrameCounter(count) {
    document.getElementById('frameCounter').textContent = count;
}

// ================ HISTORY MANAGEMENT ================

function updateHistory(historyData) {
    updateHistoryUI(historyData);
}

function updateHistoryUI(historyData) {
    const container = document.getElementById('historyContainer');
    
    if (!historyData || historyData.length === 0) {
        container.innerHTML = '<p class="text-center">No results yet</p>';
        return;
    }
    
    container.innerHTML = historyData.map((item, idx) => `
        <div class="glass history-item">
            <div class="history-header">
                <h4>Result #${idx + 1}</h4>
                <small>${new Date(item.timestamp).toLocaleTimeString()}</small>
            </div>
            
            <div class="metric">
                <span class="label">Terrain:</span>
                <span class="value terrain-${item.terrain.toLowerCase().replace(/\s+/g, '-')}">${item.terrain}</span>
            </div>
            
            <div class="metric">
                <span class="label">Confidence:</span>
                <span class="value">${item.confidence}</span>
            </div>
            
            <div class="metric">
                <span class="label">Decision:</span>
                <span class="value decision-${item.decision.replace(/\s+/g, '-').toLowerCase()}">${item.decision}</span>
            </div>
            
            <p class="description">${item.description}</p>
            
            ${item.mask ? `<img src="data:image/jpeg;base64,${item.mask}" alt="Mask" style="width: 100%; border-radius: 5px; margin-top: 10px;">` : ''}
        </div>
    `).join('');
}

function clearHistory() {
    if (confirm('Clear all history?')) {
        socket.emit('clear_history');
        updateHistoryUI([]);
    }
}

// ================ INITIALIZATION ================

window.addEventListener('load', () => {
    console.log('🚀 OFFROAD AI v2.0 loaded');
    console.log('Waiting for WebSocket connection...');
    
    // Set button event listeners
    document.getElementById('startBtn').addEventListener('click', startLiveStream);
    document.getElementById('stopBtn').addEventListener('click', stopLiveStream);
});

// ================ ERROR HANDLING ================

window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

// ================ CLEANUP ================

window.addEventListener('beforeunload', () => {
    if (isStreamingActive) {
        stopLiveStream();
    }
});
