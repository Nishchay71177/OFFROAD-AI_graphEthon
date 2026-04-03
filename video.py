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


def extract_7_frames(video_path):
    """Extract exactly 7 equally-spaced frames from a video"""
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    # Calculate frame indices for 7 equal parts
    frame_indices = []
    for i in range(7):
        frame_idx = int((i * total_frames) / 7)
        frame_indices.append(frame_idx)
    
    frames = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames