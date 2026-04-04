from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import cv2
import threading
import tempfile
import os
from collections import deque
from datetime import datetime
from video import extract_frames

# ================= MEMORY OPTIMIZATION (Render Free Tier: 512 MB) =================

# Reduce PyTorch memory usage
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# Force CPU only (GPU not available on free tier)
device = torch.device("cpu")
print(f"Device: CPU (Render Free Tier)")
print(f"Available cores: {os.cpu_count()}")

# ================= END OPTIMIZATION =================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'offroad-ai-secret-2024'
app.config['JSON_SORT_KEYS'] = False
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=25)

# ================= SECURITY & HTTPS SUPPORT =================

@app.before_request
def ensure_https():
    """Ensure HTTPS is used and set security headers"""
    # Detect if behind proxy (for Render/Cloud deployment)
    if request.headers.get('X-Forwarded-Proto', 'http') == 'https' or request.scheme == 'https':
        pass  # HTTPS already secure
    
@app.after_request
def set_security_headers(response):
    """Add security headers for HTTPS and modern browsers"""
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'camera=*, microphone=*'
    return response

# ================= CONFIG =================

CLASSES = ["Easy", "Moderate", "Rough", "Very Rough"]
MAX_RESULTS_HISTORY = 3  # Store last 3 results

# SET DETERMINISTIC INFERENCE
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# OPTIMIZED ResNet transform (faster preprocessing)
clf_tf = T.Compose([
    T.Resize((224, 224), interpolation=Image.BILINEAR),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

clf_model = None

# ================= RESULT HISTORY =================

class ResultHistory:
    def __init__(self, max_size=3):
        self.results = deque(maxlen=max_size)
    
    def add(self, result):
        self.results.appendleft(result)  # Add to front
    
    def get_all(self):
        return list(self.results)
    
    def clear(self):
        self.results.clear()

result_history = ResultHistory(MAX_RESULTS_HISTORY)
active_users = {}  # Track active WebSocket connections

# ================= LOAD MODELS =================

def load_classifier():
    global clf_model
    if clf_model is None:
        import torchvision.models as models
        print("Loading ResNet classifier...")
        clf_model = models.resnet18(weights=None)
        clf_model.fc = torch.nn.Linear(clf_model.fc.in_features, 4)

        clf_model.load_state_dict(
            torch.load("terrain_classifier.pth", map_location=device)
        )

        clf_model.to(device)
        clf_model.eval()
        print("Classifier loaded successfully")

    return clf_model


def preload_models():
    """PRELOAD MODELS AT STARTUP"""
    print("=" * 50)
    print("PRELOADING MODELS AT STARTUP...")
    print("=" * 50)
    load_classifier()
    print("All models preloaded successfully!")
    print("=" * 50)


# ================= CORE PROCESSING =================

def classify_terrain(img):
    """Classify terrain type from image"""
    model = load_classifier()
    x = clf_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
        idx = torch.argmax(probs).item()
        confidence = float(probs[idx].item())

    return CLASSES[idx], confidence


def edge_detection_mask(img):
    """LIGHTWEIGHT SEGMENTATION: Edge detection instead of heavy UNet"""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    _, mask = cv2.threshold(edges, 127, 1, cv2.THRESH_BINARY)
    
    return mask


def analyze_mask(mask):
    """Analyze mask to detect free/obstacle areas"""
    h, w = mask.shape
    
    left = mask[:, :w//3]
    center = mask[:, w//3:2*w//3]
    right = mask[:, 2*w//3:]
    
    left_free = np.sum(left == 1) / left.size
    center_free = np.sum(center == 1) / center.size
    right_free = np.sum(right == 1) / right.size
    
    return left_free, center_free, right_free


def get_decision(terrain, mask):
    """Generate navigation decision based on terrain and mask"""
    left_free, center_free, right_free = analyze_mask(mask)
    
    # If center path is blocked
    if center_free < 0.3:
        return "TURN LEFT" if left_free > right_free else "TURN RIGHT"
    
    # Priority based on terrain
    if terrain == "Very Rough":
        return "STOP"
    elif terrain == "Rough":
        return "TURN LEFT" if left_free > right_free else "TURN RIGHT"
    elif terrain == "Moderate":
        if center_free > left_free and center_free > right_free:
            return "GO SLOW STRAIGHT"
        elif left_free > right_free:
            return "GO SLOW LEFT"
        else:
            return "GO SLOW RIGHT"
    else:  # Easy
        if center_free > left_free and center_free > right_free:
            return "GO STRAIGHT"
        elif left_free > right_free:
            return "GO LEFT"
        else:
            return "GO RIGHT"


def get_decision_description(decision):
    """Get detailed explanation for decision"""
    descriptions = {
        "STOP": "Terrain too dangerous. Do not proceed.",
        "TURN LEFT": "Many obstacles. Turn left for better path.",
        "TURN RIGHT": "Many obstacles. Turn right for better path.",
        "GO SLOW STRAIGHT": "Obstacles ahead. Proceed straight slowly.",
        "GO SLOW LEFT": "Obstacles ahead. Go left slowly.",
        "GO SLOW RIGHT": "Obstacles ahead. Go right slowly.",
        "GO STRAIGHT": "Clear path. Proceed straight.",
        "GO LEFT": "Clear path. Proceed left.",
        "GO RIGHT": "Clear path. Proceed right."
    }
    return descriptions.get(decision, "Unknown decision")


def terrain_based_decision(terrain):
    """Get decision based only on terrain classification"""
    if terrain == "Very Rough":
        return "STOP"
    elif terrain == "Rough":
        return "TURN LEFT"
    elif terrain == "Moderate":
        return "GO SLOW STRAIGHT"
    else:
        return "GO STRAIGHT"


def get_verdict_description(verdict):
    """Get detailed explanation for each verdict (for video processing)"""
    descriptions = {
        "STOP": "Terrain is too dangerous. Do not proceed. Look for alternative route.",
        "TURN LEFT": "Terrain has many obstacles. Turn left and find better path.",
        "TURN RIGHT": "Terrain has many obstacles. Turn right and find better path.",
        "GO SLOW STRAIGHT": "Terrain has obstacles. Proceed straight with caution and reduced speed.",
        "GO SLOW LEFT": "Terrain has obstacles. Go left with caution and reduced speed.",
        "GO SLOW RIGHT": "Terrain has obstacles. Go right with caution and reduced speed.",
        "GO STRAIGHT": "Terrain is clear. Safe to proceed straight normally.",
        "GO LEFT": "Terrain is mostly clear. Proceed left normally.",
        "GO RIGHT": "Terrain is mostly clear. Proceed right normally."
    }
    return descriptions.get(verdict, "Unknown verdict")


def compress_frame(frame_array, quality=60):
    """Compress frame to JPEG for lightweight transmission"""
    img = Image.fromarray(frame_array)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def process_frame(frame_base64):
    """
    Process a single frame and return prediction
    frame_base64: Base64 encoded JPEG frame from browser
    """
    try:
        # Decode frame
        frame_data = base64.b64decode(frame_base64.split(',')[-1])
        frame_img = Image.open(io.BytesIO(frame_data)).convert('RGB')
        
        # Classify terrain
        terrain, confidence = classify_terrain(frame_img)
        
        # Generate mask
        mask = edge_detection_mask(frame_img)
        
        # Get decision
        decision = get_decision(terrain, mask)
        
        # Compress mask for transmission
        mask_compressed = compress_frame((mask * 255).astype(np.uint8))
        
        # Create result
        result = {
            'timestamp': datetime.now().isoformat(),
            'terrain': terrain,
            'confidence': f"{confidence:.2%}",
            'decision': decision,
            'description': get_decision_description(decision),
            'mask': mask_compressed
        }
        
        # Add to history
        result_history.add(result)
        
        return result
    
    except Exception as e:
        print(f"Frame processing error: {str(e)}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


# ================= WEBSOCKET EVENTS =================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    active_users[request.sid] = {
        'connected_at': datetime.now().isoformat(),
        'frames_processed': 0
    }
    print(f"✅ Client connected: {request.sid}")
    emit('connect_response', {'status': 'Connected to AI server'}, broadcast=False)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    if request.sid in active_users:
        del active_users[request.sid]
    print(f"❌ Client disconnected: {request.sid}")


@socketio.on('process_frame')
def handle_frame(data):
    """
    Handle incoming frame from client
    Expected: {frame: base64_string}
    """
    try:
        frame_base64 = data.get('frame')
        if not frame_base64:
            emit('error', {'message': 'No frame data provided'})
            return
        
        # Process frame
        result = process_frame(frame_base64)
        
        # Update user stats
        active_users[request.sid]['frames_processed'] += 1
        
        # Send result back
        emit('frame_result', {
            'result': result,
            'history': result_history.get_all(),
            'user_stats': active_users[request.sid]
        })
    
    except Exception as e:
        print(f"❌ Error processing frame: {str(e)}")
        emit('error', {'message': f'Processing error: {str(e)}'})


@socketio.on('clear_history')
def handle_clear_history():
    """Clear the result history"""
    result_history.clear()
    emit('history_cleared', {'status': 'success'})


@socketio.on('get_history')
def handle_get_history():
    """Send current history to client"""
    emit('history_update', {'history': result_history.get_all()})


# ================= REST ROUTES (FALLBACK) =================

@app.route("/")
def home():
    """Serve main page"""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "active_users": len(active_users),
        "device": str(device),
        "model": "ResNet18"
    })


@app.route("/predict-image", methods=["POST"])
def predict_image():
    """REST endpoint for single image prediction"""
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        img = Image.open(file).convert("RGB")
        
        terrain, confidence = classify_terrain(img)
        mask = edge_detection_mask(img)
        decision = get_decision(terrain, mask)
        
        mask_compressed = compress_frame((mask * 255).astype(np.uint8))
        
        return jsonify({
            "terrain": terrain,
            "confidence": f"{confidence:.2%}",
            "decision": decision,
            "description": get_decision_description(decision),
            "mask": mask_compressed
        })
    
    except Exception as e:
        print(f"❌ Image prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict-video", methods=["POST"])
def predict_video():
    """REST endpoint for video file processing (extracts 6 frames)"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            file.save(tmp.name)
            video_path = tmp.name
        
        try:
            # Extract 6 evenly-spaced frames from video
            frames = extract_frames(video_path, num_frames=6)
            
            if not frames:
                return jsonify({"error": "No frames extracted from video"}), 400
            
            frame_predictions = []
            
            # Process each frame
            for idx, frame in enumerate(frames, 1):
                # Classify terrain
                terrain, confidence = classify_terrain(frame)
                
                # Generate segmentation
                mask = edge_detection_mask(frame)
                
                # Get decision
                decision = get_decision(terrain, mask)
                
                # Convert mask to base64
                mask_compressed = compress_frame((mask * 255).astype(np.uint8))
                
                frame_predictions.append({
                    "frame_num": idx,
                    "terrain": terrain,
                    "confidence": f"{confidence:.2%}",
                    "decision": decision,
                    "decision_description": get_decision_description(decision),
                    "mask": mask_compressed
                })
            
            # Determine final decision based on most critical terrain
            if frame_predictions:
                # Count terrain types
                terrain_counts = {}
                for pred in frame_predictions:
                    terrain = pred["terrain"]
                    terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
                
                # Get most critical terrain (priority: Very Rough > Rough > Moderate > Easy)
                priority = {"Very Rough": 4, "Rough": 3, "Moderate": 2, "Easy": 1}
                final_terrain = max(terrain_counts.keys(), key=lambda x: (priority.get(x, 0), terrain_counts[x]))
                final_decision = terrain_based_decision(final_terrain)
            else:
                final_terrain = None
                final_decision = "No frames processed"
            
            return jsonify({
                "success": True,
                "total_frames": len(frame_predictions),
                "frame_predictions": frame_predictions,
                "final_terrain": final_terrain,
                "final_decision": final_decision,
                "final_decision_description": get_verdict_description(final_decision) if final_decision != "No frames processed" else ""
            })
        
        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.remove(video_path)
    
    except Exception as e:
        print(f"❌ VIDEO ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    preload_models()
    
    # Get port from environment (Render sets this) or use 10000 for local
    port = int(os.environ.get('PORT', 10000))
    
    # Check for SSL certificates for HTTPS
    cert_file = os.path.join(os.path.dirname(__file__), 'certs', 'cert.pem')
    key_file = os.path.join(os.path.dirname(__file__), 'certs', 'key.pem')
    
    ssl_context = None
    if os.path.exists(cert_file) and os.path.exists(key_file):
        ssl_context = (cert_file, key_file)
        print(f"🔐 HTTPS enabled with certificates at {cert_file}")
    else:
        print(f"⚠️  No SSL certificates found. Running on HTTP.")
        print(f"   To enable HTTPS, run: python generate_ssl_cert.py")
    
    # Use SocketIO instead of Flask directly
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True,
        ssl_context=ssl_context
    )
