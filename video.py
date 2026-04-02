import cv2
from PIL import Image

def sample_video_frames(video_path, frame_skip=10, max_frames=5):

    cap = cv2.VideoCapture(video_path)

    frames = []

    idx = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        if idx % frame_skip == 0:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(Image.fromarray(frame_rgb))

            if len(frames) >= max_frames:
                break

        idx += 1

    cap.release()

    return frames