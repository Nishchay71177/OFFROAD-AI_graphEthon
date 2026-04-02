
import cv2
from PIL import Image
import math

def sample_video_frames(video_path, frame_skip=None, max_frames=None):
    """
    Sample frames from video based on FPS.
    Frame skip is calculated as 1/10 of FPS (rounded to nearest whole number).
    max_frames is auto-calculated based on video length if not provided.
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame skip: 1/10 of FPS rounded to nearest whole number
    if frame_skip is None:
        frame_skip = max(1, round(fps / 10))
    
    # Calculate max frames to sample: 1/10 of total frames rounded to nearest whole number
    if max_frames is None:
        max_frames = max(1, round(total_frames / 10))

    frames = []
    frame_indices = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            frame_indices.append(idx)

            if len(frames) >= max_frames:
                break

        idx += 1

    cap.release()

    return {
        'frames': frames,
        'frame_indices': frame_indices,
        'fps': fps,
        'total_frames': total_frames,
        'frame_skip': frame_skip,
        'sampled_count': len(frames),
        'calculated_fps': fps / 10  # 1/10 of actual FPS
    }
