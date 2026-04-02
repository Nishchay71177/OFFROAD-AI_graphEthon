from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from PIL import Image
import tempfile
import io
import base64

import segmentation_models_pytorch as smp

from video import sample_video_frames

app = Flask(__name__)
CORS(app)

# ================= LOAD MODELS =================

clf_model = models.resnet18(weights=None)
clf_model.fc = torch.nn.Linear(clf_model.fc.in_features, 4)

clf_model.load_state_dict(
    torch.load("terrain_classifier.pth", map_location="cpu")
)

clf_model.eval()

unet_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes=1,
    activation=None
)

unet_model.eval()

# ================= TRANSFORMS =================

clf_tf = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

seg_tf = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

CLASSES = ["Easy","Moderate","Rough","Very Rough"]

# ================= CORE LOGIC =================

def classify_terrain(img):

    x = clf_tf(img).unsqueeze(0)

    with torch.no_grad():

        probs = F.softmax(clf_model(x), dim=1)[0]

        idx = torch.argmax(probs).item()

    return CLASSES[idx], float(probs[idx]*100)


def unet_segment(img):

    x = seg_tf(img).unsqueeze(0)

    with torch.no_grad():

        logits = unet_model(x)

        probs = torch.sigmoid(logits)

        probs = probs.squeeze().cpu().numpy()

    mask = (probs > 0.5).astype(np.uint8)

    return mask


def split_zones(mask):

    h,w = mask.shape

    return (
        mask[:, :w//3],
        mask[:, w//3:2*w//3],
        mask[:, 2*w//3:]
    )


def free_ratio(zone):

    return np.sum(zone==1) / zone.size


def navigation_decision(mask, terrain_label):

    left,front,right = split_zones(mask)

    lf,ff,rf = free_ratio(left),free_ratio(front),free_ratio(right)

    if terrain_label == "Very Rough":

        decision = "STOP"

    elif ff < 0.3:

        decision = "TURN LEFT" if lf > rf else "TURN RIGHT"

    else:

        decision = "GO STRAIGHT"

    return decision


# ================= ROUTES =================

@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict-image", methods=["POST"])
def predict_image():

    file = request.files["file"]

    img = Image.open(file).convert("RGB")

    terrain, conf = classify_terrain(img)

    mask = unet_segment(img)

    decision = navigation_decision(mask, terrain)

    # convert mask to base64
    mask_img = Image.fromarray(mask*255)

    buffered = io.BytesIO()

    mask_img.save(buffered, format="PNG")

    mask_base64 = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({

        "terrain": terrain,

        "confidence": round(conf,2),

        "decision": decision,

        "mask": mask_base64
    })


@app.route("/predict-video", methods=["POST"])
def predict_video():

    file = request.files["file"]

    temp = tempfile.NamedTemporaryFile(delete=False)

    file.save(temp.name)

    frames = sample_video_frames(temp.name)

    decisions = []

    for frame in frames:

        terrain,_ = classify_terrain(frame)

        mask = unet_segment(frame)

        decision = navigation_decision(mask, terrain)

        decisions.append(decision)

    final_decision = max(set(decisions), key=decisions.count)

    return jsonify({

        "frame_predictions": decisions,

        "final_decision": final_decision
    })


if __name__ == "__main__":

    app.run()