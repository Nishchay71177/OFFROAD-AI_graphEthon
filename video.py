import cv2
from PIL import Image

def stream_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames (performance boost)
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yield Image.fromarray(frame_rgb)

    cap.release()