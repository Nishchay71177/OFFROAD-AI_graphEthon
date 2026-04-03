from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import cv2
import tempfile
import os
from video import extract_7_frames

app = Flask(__name__)
CORS(app)

# ================= CONFIG =================

CLASSES = ["Easy", "Moderate", "Rough", "Very Rough"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 SET DETERMINISTIC INFERENCE
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 🔥 OPTIMIZED ResNet transform (faster preprocessing)
clf_tf = T.Compose([
    T.Resize((224, 224), interpolation=Image.BILINEAR),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

clf_model = None
unet_model = None

# ================= LOAD MODELS =================

def load_classifier():
    global clf_model
    if clf_model is None:
        import torchvision.models as models
        print("📦 Loading ResNet classifier...")
        clf_model = models.resnet18(weights=None)
        clf_model.fc = torch.nn.Linear(clf_model.fc.in_features, 4)

        clf_model.load_state_dict(
            torch.load("terrain_classifier.pth", map_location=device)
        )

        clf_model.to(device)
        clf_model.eval()
        print("✅ Classifier loaded successfully")

    return clf_model


def load_unet():
    """Lightweight edge-detection based segmentation (replaces heavy UNet)"""
    return "edge_detection"


def preload_models():
    """🔥 PRELOAD MODELS AT STARTUP - eliminates lag on first request"""
    print("=" * 50)
    print("🚀 PRELOADING MODELS AT STARTUP...")
    print("=" * 50)
    load_classifier()
    print("✅ All models preloaded successfully!")
    print("=" * 50)


# ================= CORE =================

def classify_terrain(img):
    model = load_classifier()

    x = clf_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
        idx = torch.argmax(probs).item()

    return CLASSES[idx]


def unet_segment(img):
    """🔥 LIGHTWEIGHT SEGMENTATION: Use edge detection instead of heavy UNet"""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection (instant, no GPU needed)
    edges = cv2.Canny(gray, 50, 150)
    _, mask = cv2.threshold(edges, 127, 1, cv2.THRESH_BINARY)
    
    return mask


def terrain_based_decision(terrain):
    if terrain == "Very Rough":
        return "STOP"
    elif terrain == "Rough":
        return "TURN LEFT"
    elif terrain == "Moderate":
        return "GO SLOW STRAIGHT"
    else:
        return "GO STRAIGHT"


def get_verdict_description(verdict):
    """Get detailed explanation for each verdict"""
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


def analyze_mask(mask):
    """Analyze UNet mask to detect free/obstacle areas"""
    h, w = mask.shape
    
    left = mask[:, :w//3]
    center = mask[:, w//3:2*w//3]
    right = mask[:, 2*w//3:]
    
    left_free = np.sum(left == 1) / left.size
    center_free = np.sum(center == 1) / center.size
    right_free = np.sum(right == 1) / right.size
    
    return left_free, center_free, right_free


def combined_decision(terrain, mask):
    """Use BOTH ResNet (terrain) + UNet (mask) for decision"""
    
    # Analyze mask from UNet
    left_free, center_free, right_free = analyze_mask(mask)
    
    # If center path is blocked, turn
    if center_free < 0.3:
        return "TURN LEFT" if left_free > right_free else "TURN RIGHT"
    
    # If terrain is very rough, STOP
    if terrain == "Very Rough":
        return "STOP"
    elif terrain == "Rough":
        return "TURN LEFT" if left_free > right_free else "TURN RIGHT"
    elif terrain == "Moderate":
        # For moderate terrain, suggest direction based on clearest path
        if center_free > left_free and center_free > right_free:
            return "GO SLOW STRAIGHT"
        elif left_free > right_free:
            return "GO SLOW LEFT"
        else:
            return "GO SLOW RIGHT"
    else:
        # Easy terrain - suggest straight if clear, otherwise a direction
        if center_free > left_free and center_free > right_free:
            return "GO STRAIGHT"
        elif left_free > right_free:
            return "GO LEFT"
        else:
            return "GO RIGHT"


# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "running"})


@app.route("/predict-image", methods=["POST"])
def predict_image():
    try:
        file = request.files["file"]
        img = Image.open(file).convert("RGB")

        # ✅ ResNet → terrain type
        terrain = classify_terrain(img)

        # ✅ LIGHTWEIGHT segmentation → edge detection (fast!)
        mask = unet_segment(img)

        # ✅ COMBINED DECISION using terrain classification
        decision = combined_decision(terrain, mask)

        # Convert mask → base64
        mask_img = Image.fromarray(mask * 255)
        buffer = io.BytesIO()
        mask_img.save(buffer, format="PNG")

        return jsonify({
            "terrain": terrain,
            "decision": decision,
            "mask": base64.b64encode(buffer.getvalue()).decode()
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """🔥 BATCH PROCESSING: Process multiple frames efficiently"""
    try:
        files = request.files.getlist("files")
        results = []

        for file in files:
            img = Image.open(file).convert("RGB")
            terrain = classify_terrain(img)
            mask = unet_segment(img)
            decision = combined_decision(terrain, mask)

            mask_img = Image.fromarray(mask * 255)
            buffer = io.BytesIO()
            mask_img.save(buffer, format="PNG")

            results.append({
                "terrain": terrain,
                "decision": decision,
                "mask": base64.b64encode(buffer.getvalue()).decode()
            })

        return jsonify({"results": results})

    except Exception as e:
        print("❌ BATCH ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/predict-video", methods=["POST"])
def predict_video():
    """🎬 VIDEO PROCESSING: Divide video into 7 parts and analyze each"""
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
            # Extract exactly 7 frames from video
            frames = extract_7_frames(video_path)
            
            if not frames:
                return jsonify({"error": "No frames extracted from video"}), 400
            
            frame_predictions = []
            
            # Process each of the 7 frames
            for idx, frame in enumerate(frames, 1):
                # Classify terrain
                terrain = classify_terrain(frame)
                
                # Generate segmentation
                mask = unet_segment(frame)
                
                # Get decision
                decision = combined_decision(terrain, mask)
                
                # Convert mask to base64
                mask_img = Image.fromarray(mask * 255)
                buffer = io.BytesIO()
                mask_img.save(buffer, format="PNG")
                mask_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                frame_predictions.append({
                    "part": idx,
                    "terrain": terrain,
                    "decision": decision,
                    "decision_description": get_verdict_description(decision),
                    "mask": mask_base64
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
                "total_parts": len(frame_predictions),
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
        print("❌ VIDEO ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ================= RUN =================

if __name__ == "__main__":
    # Preload models before starting server
    preload_models()
    
    import os

app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 10000)),
    debug=False
)
