from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
import tempfile
import io
import base64
import os

from video import sample_video_frames

app = Flask(__name__)
CORS(app)

# ================= CONFIG =================

CLASSES = ["Easy", "Moderate", "Rough", "Very Rough"]

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

# ================= LAZY LOAD MODELS =================

clf_model = None
unet_model = None


def load_classifier():

    global clf_model

    if clf_model is None:

        import torchvision.models as models

        clf_model = models.resnet18(weights=None)

        clf_model.fc = torch.nn.Linear(
            clf_model.fc.in_features,
            4
        )

        clf_model.load_state_dict(
            torch.load(
                "terrain_classifier.pth",
                map_location="cpu"
            )
        )

        clf_model.eval()

        print("Classifier loaded")

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

        unet_model.eval()

        print("UNet loaded")

    return unet_model


# ================= CORE LOGIC =================


def classify_terrain(img):

    model = load_classifier()

    x = clf_tf(img).unsqueeze(0)

    with torch.no_grad():

        probs = F.softmax(model(x), dim=1)[0]

        idx = torch.argmax(probs).item()

    return CLASSES[idx], float(probs[idx] * 100)



def unet_segment(img):

    model = load_unet()

    x = seg_tf(img).unsqueeze(0)

    with torch.no_grad():

        logits = model(x)

        probs = torch.sigmoid(logits)

        probs = probs.squeeze().cpu().numpy()

    mask = (probs > 0.5).astype(np.uint8)

    return mask



def split_zones(mask):

    h, w = mask.shape

    return (

        mask[:, :w//3],

        mask[:, w//3:2*w//3],

        mask[:, 2*w//3:]

    )



def free_ratio(zone):

    return np.sum(zone == 1) / zone.size



def navigation_decision(mask, terrain_label):

    left, front, right = split_zones(mask)

    lf = free_ratio(left)

    ff = free_ratio(front)

    rf = free_ratio(right)

    if terrain_label == "Very Rough":

        decision = "STOP"

    elif ff < 0.30:

        decision = "TURN LEFT" if lf > rf else "TURN RIGHT"

    else:

        decision = "GO STRAIGHT"

    return decision


# ================= ROUTES =================


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/health")
def health():

    return jsonify({

        "status": "running"

    })


@app.route("/predict-image", methods=["POST"])
def predict_image():

    file = request.files["file"]

    img = Image.open(file).convert("RGB")

    terrain, conf = classify_terrain(img)

    mask = unet_segment(img)

    decision = navigation_decision(mask, terrain)

    # convert mask → base64 image

    mask_img = Image.fromarray(mask * 255)

    buffered = io.BytesIO()

    mask_img.save(buffered, format="PNG")

    mask_base64 = base64.b64encode(

        buffered.getvalue()

    ).decode()

    return jsonify({

        "terrain": terrain,

        "confidence": round(conf, 2),

        "decision": decision,

        "mask": mask_base64

    })


@app.route("/predict-video", methods=["POST"])
def predict_video():

    file = request.files["file"]

    temp = tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".mp4"

    )

    file.save(temp.name)

    video_data = sample_video_frames(

        temp.name,

        num_frames=5

    )

    frames = video_data["frames"]

    decisions = []

    for frame in frames:

        terrain, _ = classify_terrain(frame)

        mask = unet_segment(frame)

        decision = navigation_decision(

            mask,

            terrain

        )

        decisions.append(decision)

    final_decision = max(

        set(decisions),

        key=decisions.count

    )

    return jsonify({

        "decisions": decisions,

        "final_decision": final_decision

    })


# ================= RUN SERVER =================


if __name__ == "__main__":

    PORT = int(

        os.environ.get(

            "PORT",

            10000

        )

    )

    # HTTPS enabled for live camera testing

    app.run(

        host="0.0.0.0",

        port=PORT,

        ssl_context="adhoc",

        debug=True

    )