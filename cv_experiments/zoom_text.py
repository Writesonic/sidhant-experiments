import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from moviepy.editor import VideoFileClip, TextClip

# --- Configuration Helpers ---
def lerp(start, end, p):
    return start + (end - start) * p

# --- 1. Advanced Face Tracking (Position + Size) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

def get_face_data(video_path):
    """
    Returns a list of tuples: (nose_x, nose_y, face_width, face_height)
    We track size so we know where NOT to put the text.
    """
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    face_data = [] # Stores (cx, cy, width, height)
    frame_idx = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]

            # Key Landmarks (same indices as legacy API)
            nose = landmarks[4]       # Center
            chin = landmarks[152]     # Bottom
            forehead = landmarks[10]  # Top
            left_cheek = landmarks[234]  # Left (screen left)
            right_cheek = landmarks[454] # Right (screen right)

            # Coordinates
            cx, cy = int(nose.x * w), int(nose.y * h)

            # Calculate Face Dimensions (in pixels)
            face_h = int(abs(chin.y - forehead.y) * h)
            face_w = int(abs(right_cheek.x - left_cheek.x) * w)

            face_data.append((cx, cy, face_w, face_h))
        else:
            # Fallback: use previous data or center
            if face_data:
                face_data.append(face_data[-1])
            else:
                face_data.append((w//2, h//2, 100, 100)) # Default dummy

    cap.release()
    landmarker.close()
    return face_data, fps, (w, h)

def smooth_data(data, alpha=0.1):
    """Smooths X, Y, Width, and Height to prevent jitter."""
    smoothed = []
    # Initialize with first frame
    curr_state = list(data[0]) 
    
    for frame in data:
        new_state = []
        for i in range(4): # x, y, w, h
            val = (alpha * frame[i]) + ((1 - alpha) * curr_state[i])
            new_state.append(val)
        curr_state = new_state
        smoothed.append(tuple(map(int, curr_state)))
        
    return smoothed

# --- 2. The Main Effect Logic ---

def create_zoom_follow_effect(
    input_path,
    output_path,
    zoom_max=1.5,
    t_start=0,
    t_end=5,
    face_side="right",
    text_config=None
):
    """
    face_side: which side the person moves to ('left' or 'right')
    text_config = {
        "content": "HELLO!",
        "position": "left", # where the text appears ('left', 'right', 'top', 'bottom')
        "color": "white"
    }
    """
    
    print("1. Analyzing Face Trajectory...")
    raw_data, fps, (w, h) = get_face_data(input_path)
    # Alpha 0.05 is very smooth, good for zooms
    face_data = smooth_data(raw_data, alpha=0.05) 
    print(f"   Analyzed {len(face_data)} frames.")

    clip = VideoFileClip(input_path)

    # -- Pre-render text image for world-space compositing --
    text_rgb = None
    text_alpha = None
    if text_config:
        content = text_config.get('content', "Text")
        txt = TextClip(content, fontsize=70, color=text_config.get('color', 'white'), font='Arial-Bold')
        text_rgb = txt.get_frame(0)  # (H, W, 3) uint8
        text_alpha = txt.mask.get_frame(0) if txt.mask else np.ones(text_rgb.shape[:2])
        print(f"   Text size: {text_rgb.shape[1]}x{text_rgb.shape[0]}")

    # -- Helper to get face state at specific timestamp --
    def get_state(t):
        idx = int(t * fps)
        idx = max(0, min(idx, len(face_data) - 1))
        return face_data[idx] # (x, y, w, h)

    def burn_text(frame, fx, fy, fw, fh, opacity):
        """Composite text onto frame in world space, relative to face."""
        if text_rgb is None or opacity <= 0:
            return frame

        pos_mode = text_config.get('position', 'left')
        margin = text_config.get('margin', 1.3)
        th, tw = text_rgb.shape[:2]

        if pos_mode == 'left':
            tx = fx - int(fw / 2 * margin) - tw
            ty = fy - th // 2
        elif pos_mode == 'right':
            tx = fx + int(fw / 2 * margin)
            ty = fy - th // 2
        elif pos_mode == 'top':
            tx = fx - tw // 2
            ty = fy - int(fh / 2 * margin) - th
        else:  # bottom
            tx = fx - tw // 2
            ty = fy + int(fh / 2 * margin)

        # Clip to frame bounds
        x1, y1 = max(0, tx), max(0, ty)
        x2, y2 = min(w, tx + tw), min(h, ty + th)
        if x1 >= x2 or y1 >= y2:
            return frame

        sx1, sy1 = x1 - tx, y1 - ty
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

        alpha = (text_alpha[sy1:sy2, sx1:sx2, np.newaxis] * opacity).astype(np.float32)
        result = frame.copy()
        roi = result[y1:y2, x1:x2].astype(np.float32)
        txt_roi = text_rgb[sy1:sy2, sx1:sx2].astype(np.float32)
        result[y1:y2, x1:x2] = (txt_roi * alpha + roi * (1 - alpha)).astype(np.uint8)
        return result

    # -- The Video Transform --
    def process_frame(get_frame, t):
        frame = get_frame(t)

        # 1. Calculate Interpolation Factor (p)
        if t < t_start:
            p = 0
        elif t > t_end:
            p = 1
        else:
            p = (t - t_start) / (t_end - t_start)

        # Ease-in-out curve
        p = p * p * (3 - 2 * p)

        # 2. Get Current Targets
        current_zoom = lerp(1.0, zoom_max, p)
        fx, fy, fw, fh = get_state(t)

        # 3. Burn text into frame BEFORE warp (world-space sticker)
        if text_config and t_start <= t <= t_end:
            frame = burn_text(frame, fx, fy, fw, fh, opacity=p)

        # 4. Camera transform
        target_x = lerp(w/2, fx, p)
        target_y = lerp(h/2, fy, p)

        if face_side == "left":
            face_dest_x = lerp(w/2, w * 0.28, p)
        else:
            face_dest_x = lerp(w/2, w * 0.72, p)
        face_dest_y = h/2

        shift_x = face_dest_x - (target_x * current_zoom)
        shift_y = face_dest_y - (target_y * current_zoom)

        M = np.float32([
            [current_zoom, 0, shift_x],
            [0, current_zoom, shift_y]
        ])

        # 5. Warp foreground
        foreground = cv2.warpAffine(frame, M, (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)

        # 6. Fade to black from the side opposite the face
        gradient = np.ones((h, w), dtype=np.float32)
        fade_width = int(w * 0.35)
        ramp = np.linspace(0, 1, fade_width)
        if face_side == "right":
            # Face goes right, fade to black on the left
            gradient[:, :fade_width] = ramp[np.newaxis, :]
        else:
            # Face goes left, fade to black on the right
            gradient[:, w - fade_width:] = ramp[::-1][np.newaxis, :]

        # Only apply fade as the effect progresses
        fade_alpha = (1 - p) + p * gradient
        final = (foreground.astype(np.float32) * fade_alpha[:, :, np.newaxis]).astype(np.uint8)
        return final

    video_layer = clip.fl(process_frame)
    video_layer.write_videofile(output_path, fps=24)

# --- Usage Example ---

create_zoom_follow_effect(
    input_path="vid.mp4",
    output_path=f"output_{int(time.time())}.mp4",
    zoom_max=1.1,
    t_start=1.0,
    t_end=6.0,
    face_side="right",     # person pans to the right
    text_config={
        "content": "Wait for it...",
        "position": "left",  # text appears on the left
        "color": "yellow"
    }
)