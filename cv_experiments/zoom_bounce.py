"""
Zoom-Bounce — Dramatic Punch-In / Punch-Out Face Tracking Effect
=================================================================
Face-tracking zoom that punches IN then back OUT (bell curve),
with multiple bounce windows and selectable easing modes.

Based on opte.py pipeline: MediaPipe face tracking → affine warp →
per-row edge-fade → overlay composite → FFmpeg encode.
"""

import os
import queue
import subprocess
import threading
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from moviepy.editor import TextClip, VideoFileClip

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

def lerp(a, b, t):
    return a + (b - a) * t


# ─── Bounce easing functions ────────────────────────────────────────────────
# All take an array of normalized time [0,1] and return intensity [0,1] (bell)

def ease_smooth(t):
    """sin(pi*t) — clean symmetric punch in then out."""
    return np.sin(np.pi * t)

def ease_snap(t):
    """Fast attack, brief hold at peak, fast release — editorial punch."""
    out = np.zeros_like(t, dtype=np.float32)
    attack = t < 0.25
    hold = (t >= 0.25) & (t <= 0.75)
    release = t > 0.75
    out[attack] = (t[attack] / 0.25) ** 2
    out[hold] = 1.0
    r = (t[release] - 0.75) / 0.25
    out[release] = 1.0 - r ** 2
    return out

def ease_overshoot(t):
    """Elastic — overshoots ~15%, springs back, then releases. Playful energy."""
    base = np.sin(np.pi * t)
    overshoot = 0.15 * np.sin(2.0 * np.pi * t)
    return np.clip(base + overshoot, 0.0, 1.15)

EASE_FUNCTIONS = {
    "smooth": ease_smooth,
    "snap": ease_snap,
    "overshoot": ease_overshoot,
}

# ─── Attack-only easing (0→1) for "in" events ──────────────────────────────

def ease_in_smooth(t):
    return np.sin(np.pi / 2 * t)

def ease_in_snap(t):
    out = np.zeros_like(t, dtype=np.float32)
    fast = t < 0.5
    out[fast] = (t[fast] / 0.5) ** 2
    out[~fast] = 1.0
    return out

def ease_in_overshoot(t):
    base = np.sin(np.pi / 2 * t)
    overshoot = 0.15 * np.sin(np.pi * t)
    return np.clip(base + overshoot, 0.0, 1.15)

EASE_IN_FUNCTIONS = {
    "smooth": ease_in_smooth,
    "snap": ease_in_snap,
    "overshoot": ease_in_overshoot,
}

# ─── Release-only easing (1→0) for "out" events ────────────────────────────

def ease_out_smooth(t):
    return np.cos(np.pi / 2 * t)

def ease_out_snap(t):
    return np.clip(1.0 - t ** 2, 0.0, 1.0)

def ease_out_overshoot(t):
    base = np.cos(np.pi / 2 * t)
    undershoot = -0.15 * np.sin(np.pi * t)
    return np.clip(base + undershoot, 0.0, 1.15)

EASE_OUT_FUNCTIONS = {
    "smooth": ease_out_smooth,
    "snap": ease_out_snap,
    "overshoot": ease_out_overshoot,
}


def _parse_events(bounces, default_mode, default_zoom):
    """
    Normalize bounce entries (tuples and dicts) into canonical event list.

    Returns list of dicts: {"action", "start", "end", "ease", "zoom"}
    Validates: no double zoom-in, no zoom-out without prior zoom-in.
    """
    events = []
    for b in bounces:
        if isinstance(b, dict):
            ev = {
                "action": b["action"],
                "start": b["start"],
                "end": b["end"],
                "ease": b.get("ease", default_mode),
                "zoom": b.get("zoom", default_zoom),
            }
            events.append(ev)
        else:
            # Legacy tuple: (start, end[, mode[, zoom]])
            if len(b) == 2:
                ev = {"action": "bounce", "start": b[0], "end": b[1],
                      "ease": default_mode, "zoom": default_zoom}
            elif len(b) == 3:
                ev = {"action": "bounce", "start": b[0], "end": b[1],
                      "ease": b[2], "zoom": default_zoom}
            else:
                ev = {"action": "bounce", "start": b[0], "end": b[1],
                      "ease": b[2], "zoom": b[3]}
            events.append(ev)

    # Validate in/out pairing
    zoomed_in = False
    last_zoom = default_zoom
    for ev in events:
        action = ev["action"]
        if action == "in":
            if zoomed_in:
                raise ValueError(f"Double zoom-in at t={ev['start']}: already zoomed in")
            zoomed_in = True
            last_zoom = ev["zoom"]
        elif action == "out":
            if not zoomed_in:
                raise ValueError(f"Zoom-out at t={ev['start']} without prior zoom-in")
            zoomed_in = False
            # Out inherits zoom from its paired in
            ev["zoom"] = last_zoom
        # "bounce" is self-contained, no state change

    return events


def build_bounce_curves(n_frames, fps, bounces, default_mode, default_zoom):
    """
    Build per-frame p (intensity 0-1) and zoom arrays.

    bounces: list of tuples or dicts (see _parse_events).
    Returns: (times, p_curve, zooms) — all shape (n_frames,) float32
    """
    times = np.arange(n_frames, dtype=np.float32) / fps
    p_curve = np.zeros(n_frames, dtype=np.float32)
    zooms = np.ones(n_frames, dtype=np.float32)

    events = _parse_events(bounces, default_mode, default_zoom)

    # Process each event
    for ev in events:
        action = ev["action"]
        bs, be = ev["start"], ev["end"]
        mode = ev["ease"]
        zm = ev["zoom"]
        dur = max(be - bs, 1e-9)
        mask = (times >= bs) & (times <= be)

        if action == "bounce":
            ease_fn = EASE_FUNCTIONS[mode]
            t_norm = np.clip((times[mask] - bs) / dur, 0.0, 1.0)
            p_vals = ease_fn(t_norm)
            new_zooms = 1.0 + (zm - 1.0) * p_vals
            better = new_zooms > zooms[mask]
            p_curve[mask] = np.where(better, p_vals, p_curve[mask])
            zooms[mask] = np.where(better, new_zooms, zooms[mask])

        elif action == "in":
            ease_fn = EASE_IN_FUNCTIONS[mode]
            t_norm = np.clip((times[mask] - bs) / dur, 0.0, 1.0)
            p_vals = ease_fn(t_norm)
            p_curve[mask] = np.maximum(p_curve[mask], p_vals)
            zooms[mask] = np.maximum(zooms[mask], 1.0 + (zm - 1.0) * p_vals)

        elif action == "out":
            ease_fn = EASE_OUT_FUNCTIONS[mode]
            t_norm = np.clip((times[mask] - bs) / dur, 0.0, 1.0)
            p_vals = ease_fn(t_norm)
            p_curve[mask] = np.maximum(p_curve[mask], p_vals)
            zooms[mask] = np.maximum(zooms[mask], 1.0 + (zm - 1.0) * p_vals)

    # Fill holds: between each in-end and its paired out-start, hold at p=1.0
    zoomed_in = False
    in_end = 0.0
    in_zoom = default_zoom
    for ev in events:
        if ev["action"] == "in":
            zoomed_in = True
            in_end = ev["end"]
            in_zoom = ev["zoom"]
        elif ev["action"] == "out" and zoomed_in:
            out_start = ev["start"]
            hold_mask = (times > in_end) & (times < out_start)
            p_curve[hold_mask] = 1.0
            zooms[hold_mask] = in_zoom
            zoomed_in = False

    return times, p_curve, zooms


# ─── Encoder detection ───────────────────────────────────────────────────────

def _probe_encoder(name):
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=black:s=64x64:d=0.04",
             "-c:v", name, "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return r.returncode == 0
    except Exception:
        return False

def detect_best_encoder():
    if not hasattr(detect_best_encoder, "_c"):
        for e in ["h264_nvenc", "h264_videotoolbox", "h264_qsv", "libx264"]:
            if _probe_encoder(e):
                detect_best_encoder._c = e
                break
        else:
            detect_best_encoder._c = "libx264"
        print(f"   Encoder: {detect_best_encoder._c}")
    return detect_best_encoder._c


# ─── Overlay ─────────────────────────────────────────────────────────────────

class Overlay:
    def get_frame(self, t):
        raise NotImplementedError

class TextOverlay(Overlay):
    def __init__(self, content, color='white', fontsize=80, font='Arial-Bold',
                 max_width=None, max_height=None):
        fontsize = self._fit(content, fontsize, color, font, max_width, max_height)
        kw = dict(fontsize=fontsize, color=color, font=font)
        if max_width:
            kw.update(size=(max_width, None), method='caption')
        txt = TextClip(content, **kw)
        self.img = np.ascontiguousarray(txt.get_frame(0), dtype=np.float32)
        if txt.mask:
            m = txt.mask.get_frame(0)
            if m.max() > 1.0: m = m / 255.0
            self.mask = np.ascontiguousarray(m[:, :, np.newaxis], dtype=np.float32)
        else:
            self.mask = np.ones((*self.img.shape[:2], 1), dtype=np.float32)

    @staticmethod
    def _fit(content, fs, color, font, mw, mh):
        if not mw and not mh: return fs
        for s in range(fs, 19, -4):
            kw = dict(fontsize=s, color=color, font=font)
            if mw: kw.update(size=(mw, None), method='caption')
            sh = TextClip(content, **kw).get_frame(0).shape[:2]
            if (not mw or sh[1] <= mw) and (not mh or sh[0] <= mh): return s
        return 20

    def get_frame(self, t):
        return self.img, self.mask

class ImageOverlay(Overlay):
    def __init__(self, path):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
            self.img = np.ascontiguousarray(raw[:, :, :3], dtype=np.float32)
            self.mask = np.ascontiguousarray(raw[:, :, 3:4] / 255.0, dtype=np.float32)
        else:
            self.img = np.ascontiguousarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), dtype=np.float32)
            self.mask = np.ones((*raw.shape[:2], 1), dtype=np.float32)
    def get_frame(self, t):
        return self.img, self.mask

class ClipOverlay(Overlay):
    def __init__(self, path):
        self.clip = VideoFileClip(path, has_mask=True)
    def get_frame(self, t):
        t = min(t, self.clip.duration - 0.01)
        img = np.ascontiguousarray(self.clip.get_frame(t), dtype=np.float32)
        if self.clip.mask:
            m = self.clip.mask.get_frame(t)
            if m.max() > 1.0: m = m / 255.0
            mask = np.ascontiguousarray(m[:, :, np.newaxis], dtype=np.float32)
        else:
            mask = np.ones((*img.shape[:2], 1), dtype=np.float32)
        return img, mask

def create_overlay(cfg):
    t = cfg.get('type', 'text')
    if t == 'text':
        return TextOverlay(cfg.get('content', 'Text'), cfg.get('color', 'white'),
                           cfg.get('fontsize', 80), cfg.get('font', 'Arial-Bold'),
                           cfg.get('_avail_w'), cfg.get('_avail_h'))
    if t == 'image': return ImageOverlay(cfg['path'])
    if t == 'clip': return ClipOverlay(cfg['path'])
    raise ValueError(f"Unknown overlay type: {t}")


# ─── Threaded reader ─────────────────────────────────────────────────────────

class ThreadedVideoReader:
    def __init__(self, path, queue_size=128):
        self.cap = cv2.VideoCapture(path)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                self.stopped = True
                return
            self.q.put(f)

    def read(self):
        try:
            return (True, self.q.get(timeout=1.0))
        except queue.Empty:
            return (False, None)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()


# ─── Face detection ──────────────────────────────────────────────────────────

def get_face_data(video_path):
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    lm = vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    data, idx, default = [], 0, (w // 2, h // 2, 100, 100)
    while True:
        ok, bgr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                  int(idx * 1000 / fps))
        idx += 1
        if res.face_landmarks:
            f = res.face_landmarks[0]
            data.append((int(f[4].x * w), int(f[4].y * h),
                         int(abs(f[454].x - f[234].x) * w),
                         int(abs(f[152].y - f[10].y) * h)))
        else:
            data.append(data[-1] if data else default)
    cap.release(); lm.close()
    return data, fps, (w, h)

def smooth_data(data, alpha=0.1):
    a = np.array(data, dtype=np.float64)
    o = np.empty_like(a)
    o[0] = a[0]
    inv = 1.0 - alpha
    for i in range(1, len(a)):
        o[i] = alpha * a[i] + inv * o[i - 1]
    return o.astype(np.int32)


EDGE_STRIP_FRAC = 0.04
FADE_WIDTH_FRAC = 0.25


# ─── FFmpeg writer ───────────────────────────────────────────────────────────

def open_ffmpeg_writer(path, w, h, fps, enc):
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
           "-c:v", enc, "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if enc == "libx264":     cmd += ["-preset", "fast", "-crf", "18"]
    elif enc == "h264_nvenc": cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "20"]
    elif enc == "h264_videotoolbox": cmd += ["-q:v", "65"]
    cmd.append(path)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def mux_audio(src, silent, out):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", src],
        capture_output=True, text=True,
    )
    has_audio = probe.returncode == 0 and probe.stdout.strip() != ""

    if has_audio:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", silent, "-i", src,
             "-c:v", "copy", "-c:a", "aac",
             "-map", "0:v:0", "-map", "1:a:0",
             "-shortest", out],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return
        print(f"   Warning: Mux with audio failed (exit {result.returncode}):")
        print(f"     {result.stderr[-300:]}")
        print(f"     Falling back to video-only ...")

    subprocess.run(
        ["ffmpeg", "-y", "-i", silent, "-c:v", "copy", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not has_audio:
        print("   (source has no audio track — skipped)")


# ─── Main Effect ─────────────────────────────────────────────────────────────

def create_zoom_bounce_effect(
    input_path, output_path,
    zoom_max=1.4,
    bounces=None,
    bounce_mode="snap",
    face_side="center",
    overlay_config=None,
    text_config=None,
    fade_mode="band",
    stabilize=0.0,
    stabilize_alpha=0.02,
):
    """
    Dramatic punch-in / punch-out zoom that tracks a face.

    Args:
        input_path:     Source video file
        output_path:    Output video file
        zoom_max:       Peak zoom level (1.0 = none, 1.4 = 40% punch)
        bounces:        List of bounce tuples. Each can be:
                          (start, end) — uses global bounce_mode/zoom_max
                          (start, end, mode) — per-bounce easing override
                          (start, end, mode, zoom) — per-bounce easing + zoom
                        Defaults to [(1.0, 2.5)] if None.
        bounce_mode:    "smooth" (sin bell), "snap" (fast attack/release),
                        "overshoot" (elastic spring)
        face_side:      "center", "left", or "right" — where face lands on screen.
                        "center" keeps face centered (pure zoom, no lateral shift).
        overlay_config: Overlay dict (same as opte.py) or None
        text_config:    Deprecated alias for overlay_config
        fade_mode:      "band" (per-row edge color) or "average"
    """
    if bounces is None:
        bounces = [(1.0, 2.5)]
    if overlay_config is None and text_config is not None:
        overlay_config = text_config
    if bounce_mode not in EASE_FUNCTIONS:
        raise ValueError(f"Unknown bounce_mode: {bounce_mode!r}. Use: {list(EASE_FUNCTIONS)}")

    print("1. Analyzing face trajectory ...")
    raw_data, fps, (w, h) = get_face_data(input_path)
    face_data = smooth_data(raw_data, alpha=0.05)
    n_frames = len(face_data)
    # Heavier smoothing: used for stabilization crop AND to dampen tracking jitter when zoomed
    face_data_stable = smooth_data(raw_data, alpha=stabilize_alpha)

    # Build the bounce curves (supports per-bounce mode + zoom overrides)
    times, p_curve, zooms = build_bounce_curves(n_frames, fps, bounces, bounce_mode, zoom_max)

    # Overlay prep
    overlay = None
    if overlay_config:
        if overlay_config.get('type', 'text') == 'text':
            mfw = float(np.median(face_data[:, 2]))
            sfw = mfw * zoom_max
            mg = overlay_config.get('margin', 1.8)
            pad = int(w * 0.03)
            if face_side == "center":
                fcx = w * 0.5
            elif face_side == "right":
                fcx = w * 0.72
            else:
                fcx = w * 0.28
            pos = overlay_config.get('position', 'left')
            if pos == 'left':
                aw = int(fcx - (sfw / 2 * mg) - pad)
            elif pos == 'right':
                aw = int(w - (fcx + sfw / 2 * mg) - pad)
            else:
                aw = int(w * 0.5)
            overlay_config = {**overlay_config, '_avail_w': max(aw, 100), '_avail_h': int(h * 0.6)}
        overlay = create_overlay(overlay_config)

    # Pre-allocate buffers
    buf_warped = np.empty((h, w, 3), dtype=np.uint8)
    buf_out = np.empty((h, w, 3), dtype=np.uint8)
    buf_warped_f32 = np.empty((h, w, 3), dtype=np.float32)
    buf_fade_alpha = np.empty((h, w, 1), dtype=np.float32)
    buf_blend = np.empty((h, w, 3), dtype=np.float32)
    fade_bg_buf = np.empty((h, w, 3), dtype=np.float32)
    buf_rgb = np.empty((h, w, 3), dtype=np.uint8)
    if face_side == "center":
        dest_x_full = w * 0.5
    elif face_side == "left":
        dest_x_full = w * 0.28
    else:
        dest_x_full = w * 0.72

    # Hoist overlay config lookups
    ovl_pos = (overlay_config.get('position', 'left') if overlay_config else 'left')
    ovl_mg = (overlay_config.get('margin', 1.8) if overlay_config else 1.8)

    # Precompute which bounce window each frame belongs to (for overlay timing)
    frame_bounce_idx = np.full(n_frames, -1, dtype=np.int32)
    for bi, b in enumerate(bounces):
        if isinstance(b, dict):
            bs, be = b["start"], b["end"]
        else:
            bs, be = b[0], b[1]
        mask = (times >= bs) & (times <= be)
        frame_bounce_idx[mask] = bi

    # Gradient fade setup — not needed for "center" (zoom crops evenly)
    need_fade = face_side != "center"
    edge_strip = max(int(w * EDGE_STRIP_FRAC), 1)
    fade_width = int(w * FADE_WIDTH_FRAC)
    if need_fade:
        ramp = np.linspace(0, 1, fade_width).astype(np.float32)
        base_gradient = np.ones((h, w), dtype=np.float32)
        if face_side == "right":
            base_gradient[:, :fade_width] = ramp[np.newaxis, :]
        else:
            base_gradient[:, w - fade_width:] = ramp[::-1][np.newaxis, :]
        base_gradient_3ch = base_gradient[:, :, np.newaxis]

    # I/O
    enc = detect_best_encoder()
    tmp = output_path + ".tmp_silent.mp4"
    writer = open_ffmpeg_writer(tmp, w, h, fps, enc)
    reader = ThreadedVideoReader(input_path, queue_size=64)

    print(f"2. Processing {n_frames} frames ({bounce_mode} mode, {len(bounces)} bounce(s)) ...")
    t0 = time.monotonic()

    for idx in range(n_frames):
        ok, bgr = reader.read()
        if not ok:
            break

        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=buf_rgb)
        rgb = buf_rgb
        t = times[idx]
        p = float(p_curve[idx])
        z = float(zooms[idx])

        # ── Warp geometry ─────────────────────────────────────────────
        # Blend face position toward heavier-smoothed data as zoom
        # increases — prevents jitter amplification at high zoom
        fx_raw, fy_raw, fw_raw, fh_raw = face_data[idx]
        if face_data_stable is not None and p > 0.001:
            fx_st, fy_st = float(face_data_stable[idx][0]), float(face_data_stable[idx][1])
            fx = lerp(float(fx_raw), fx_st, p)
            fy = lerp(float(fy_raw), fy_st, p)
        else:
            fx, fy = float(fx_raw), float(fy_raw)
        fw, fh = float(fw_raw), float(fh_raw)

        if stabilize and p < 0.001:
            # Pure stabilization: subtle crop centered on heavily-smoothed face
            sfx_s, sfy_s = float(face_data_stable[idx][0]), float(face_data_stable[idx][1])
            sz = stabilize
            s_sx = w / 2 - sfx_s * sz
            s_sy = h / 2 - sfy_s * sz
            M = np.float32([[sz, 0, s_sx], [0, sz, s_sy]])
            sfx = fx * sz + s_sx
            sfy = fy * sz + s_sy
            sfw = fw * sz
            sfh = fh * sz
        elif stabilize and p > 0:
            # Blend between stabilization center and bounce center
            sfx_s, sfy_s = float(face_data_stable[idx][0]), float(face_data_stable[idx][1])
            sz_stab = stabilize
            s_sx_stab = w / 2 - sfx_s * sz_stab
            s_sy_stab = h / 2 - sfy_s * sz_stab
            sx_b = dest_x_full - fx * z
            sy_b = h / 2 - fy * z
            ez = lerp(sz_stab, z, p)
            e_sx = lerp(s_sx_stab, sx_b, p)
            e_sy = lerp(s_sy_stab, sy_b, p)
            M = np.float32([[ez, 0, e_sx], [0, ez, e_sy]])
            sfx = fx * ez + e_sx
            sfy = fy * ez + e_sy
            sfw = fw * ez
            sfh = fh * ez
        else:
            # No stabilization: existing behavior
            tx = lerp(w / 2, fx, p)
            ty = lerp(h / 2, fy, p)
            dx = lerp(w / 2, dest_x_full, p)
            sx = dx - tx * z
            sy = h / 2 - ty * z
            M = np.float32([[z, 0, sx], [0, z, sy]])
            sfx = fx * z + sx
            sfy = fy * z + sy
            sfw = fw * z
            sfh = fh * z

        # ── Warp ──────────────────────────────────────────────────────
        cv2.warpAffine(rgb, M, (w, h), dst=buf_warped,
                       borderMode=cv2.BORDER_REPLICATE)

        # ── Gradient fade ─────────────────────────────────────────────
        if p < 0.001 or not need_fade:
            # No zoom active or center mode — no edge fade needed
            np.copyto(buf_out, buf_warped)
        else:
            if face_side == "right":
                edge_band = buf_warped[:, :edge_strip].mean(axis=1, dtype=np.float32)
            else:
                edge_band = buf_warped[:, w - edge_strip:].mean(axis=1, dtype=np.float32)

            CRUSH_H = 6
            edge_col = edge_band.reshape(h, 1, 3)
            crushed = cv2.resize(edge_col, (1, CRUSH_H), interpolation=cv2.INTER_AREA)
            edge_band = cv2.resize(crushed, (1, h), interpolation=cv2.INTER_LINEAR).reshape(h, 3)
            fade_bg_buf[:] = edge_band[:, np.newaxis, :]
            fade_bg = fade_bg_buf

            buf_warped_f32[:] = buf_warped
            np.multiply(base_gradient_3ch, p, out=buf_fade_alpha)
            buf_fade_alpha += (1.0 - p)
            np.multiply(buf_warped_f32, buf_fade_alpha, out=buf_blend)
            np.subtract(1.0, buf_fade_alpha, out=buf_fade_alpha)
            np.multiply(fade_bg, buf_fade_alpha, out=buf_warped_f32)
            np.add(buf_blend, buf_warped_f32, out=buf_blend)
            np.clip(buf_blend, 0, 255, out=buf_blend)
            np.copyto(buf_out, buf_blend.astype(np.uint8))

        # ── Overlay ───────────────────────────────────────────────────
        if overlay and overlay_config and p > 0.01:
            # Overlay opacity ramps with zoom intensity
            opacity = min(p * 3.0, 1.0)
            if opacity > 0:
                oi, om = overlay.get_frame(t)
                oh, ow_ = oi.shape[:2]
                if ovl_pos == 'left':
                    ox, oy = int(sfx - sfw/2*ovl_mg - ow_), int(sfy - oh//2)
                elif ovl_pos == 'right':
                    ox, oy = int(sfx + sfw/2*ovl_mg), int(sfy - oh//2)
                elif ovl_pos == 'top':
                    ox, oy = int(sfx - ow_//2), int(sfy - sfh/2*ovl_mg - oh)
                else:
                    ox, oy = int(sfx - ow_//2), int(sfy + sfh/2*ovl_mg)

                x1, y1 = max(0, ox), max(0, oy)
                x2, y2 = min(w, ox + ow_), min(h, oy + oh)
                if x1 < x2 and y1 < y2:
                    s1, s2 = x1 - ox, y1 - oy
                    roi = buf_out[y1:y2, x1:x2].astype(np.float32)
                    o = oi[s2:s2+y2-y1, s1:s1+x2-x1]
                    a = om[s2:s2+y2-y1, s1:s1+x2-x1] * opacity
                    buf_out[y1:y2, x1:x2] = (o*a + roi*(1.0 - a)).astype(np.uint8)

        writer.stdin.write(buf_out.tobytes())
        if idx % 50 == 0:
            print(f"   frame {idx}/{n_frames}", flush=True)

    elapsed = time.monotonic() - t0
    actual = min(idx + 1, n_frames)
    print(f"   {actual} frames in {elapsed:.1f}s ({actual/max(elapsed,.01):.1f} fps)")

    reader.release()
    writer.stdin.close()
    writer.wait()

    print("3. Muxing audio ...")
    mux_audio(input_path, tmp, output_path)
    os.remove(tmp)
    print(f"Done -> {output_path}")


# ─── Usage ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = int(time.time())

    # Single video, all features: in/out holds, all 6 ease combos,
    # legacy bell-curve bounces, stabilization baseline between events
    create_zoom_bounce_effect(
        input_path="longvid.mp4",
        output_path=f"longvid_events_{ts}.mp4",
        stabilize=1.03,
        bounces=[
            # 0-1.5s: stabilization-only baseline (no events, jitter absorbed)

            # 1. snap in → long hold → smooth out
            {"action": "in",  "start": 1.5,  "end": 2.0,  "ease": "snap",      "zoom": 1.4},
            {"action": "out", "start": 4.5,  "end": 5.3,  "ease": "smooth"},

            # 5.3-6s: stabilization gap

            # 2. smooth in → short hold → snap out
            {"action": "in",  "start": 6.0,  "end": 6.8,  "ease": "smooth",    "zoom": 1.3},
            {"action": "out", "start": 8.0,  "end": 8.3,  "ease": "snap"},

            # 3. overshoot in → hold → overshoot out
            {"action": "in",  "start": 9.5,  "end": 10.0, "ease": "overshoot", "zoom": 1.5},
            {"action": "out", "start": 12.0, "end": 12.5, "ease": "overshoot"},

            # 4. legacy tuple bounce — bell curve regression test
            (14.0, 15.5, "smooth", 1.3),

            # 5. legacy snap bounce
            (16.5, 17.5, "snap", 1.5),

            # 6. legacy overshoot bounce
            (18.5, 19.5, "overshoot", 1.4),

            # 7. overshoot in → snap out (cross-ease, tight hold)
            {"action": "in",  "start": 21.0, "end": 21.4, "ease": "overshoot", "zoom": 1.6},
            {"action": "out", "start": 22.5, "end": 22.8, "ease": "snap"},

            # 8. snap in → overshoot out (cross-ease)
            {"action": "in",  "start": 24.0, "end": 24.2, "ease": "snap",      "zoom": 1.4},
            {"action": "out", "start": 26.0, "end": 26.5, "ease": "overshoot"},

            # 27-30s: stabilization-only tail (verifies clean return)
        ],
    )
