import cv2
from PIL import Image

def extract_frames(video_path, num_frames=6):
    """Extract equally-spaced frames from a video"""
    cap = cv2.VideoCapture(video_path)
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return []
        
        # Calculate frame indices for equal parts
        frame_indices = []
        for i in range(num_frames):
            frame_idx = int((i * total_frames) / num_frames)
            frame_indices.append(frame_idx)
        
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        return frames
    
    finally:
        # Automatically close the stream
        cap.release()