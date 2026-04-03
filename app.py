from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# ================= CONFIG =================

CLASSES = ["Easy", "Moderate", "Rough", "Very Rough"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 ResNet transform (with normalization)
clf_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 🔥 UNet transform
seg_tf = T.Compose([
    T.Resize((256, 256)),
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

        clf_model = models.resnet18(weights=None)
        clf_model.fc = torch.nn.Linear(clf_model.fc.in_features, 4)

        clf_model.load_state_dict(
            torch.load("terrain_classifier.pth", map_location=device)
        )

        clf_model.to(device)
        clf_model.eval()

        print("✅ Classifier loaded")

    return clf_model


def load_unet():
    global unet_model
    if unet_model is None:
        import segmentation_models_pytorch as smp

        unet_model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            classes=1,
            activation=None
        )

        # ⚠️ If you have trained weights, load here:
        # unet_model.load_state_dict(torch.load("unet.pth", map_location=device))

        unet_model.to(device)
        unet_model.eval()

        print("✅ UNet loaded")

    return unet_model


# ================= CORE =================

def classify_terrain(img):
    model = load_classifier()

    x = clf_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
        idx = torch.argmax(probs).item()

    return CLASSES[idx]


def unet_segment(img):
    model = load_unet()

    x = seg_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    return (probs > 0.5).astype(np.uint8)


def terrain_based_decision(terrain):
    if terrain == "Very Rough":
        return "STOP"
    elif terrain == "Rough":
        return "TURN LEFT"
    elif terrain == "Moderate":
        return "GO SLOW"
    else:
        return "GO STRAIGHT"


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
    
    # Terrain-based decision
    terrain_decision = terrain_based_decision(terrain)
    
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
        return "GO SLOW"
    else:
        return "GO STRAIGHT"


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

        # ✅ UNet → segmentation mask (obstacles/free paths)
        mask = unet_segment(img)

        # ✅ COMBINED DECISION using BOTH models
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


# ================= RUN =================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000,
        ssl_context="adhoc",   # 🔥 needed for camera
        debug=True
    )