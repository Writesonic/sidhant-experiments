"""
Zoom-Follow V3 — Gradient Fade + Static Lock
=============================================
Uses BORDER_REPLICATE + per-row edge-color gradient fade
(matching zoom_text.py approach) for seamless edge blending.
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
            self.q.put(f)  # blocks when full

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


EDGE_STRIP_FRAC = 0.04    # Fraction of width to sample at the edge
FADE_WIDTH_FRAC = 0.25     # Fraction of width for the gradient fade


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
    """
    Mux audio from src onto silent video. If src has no audio or mux fails,
    fall back to just copying the silent video as-is.
    """
    # First, check if source has an audio stream
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
        # Log the actual error for debugging
        print(f"   ⚠ Mux with audio failed (exit {result.returncode}):")
        print(f"     {result.stderr[-300:]}")
        print(f"     Falling back to video-only …")

    # Fallback: just remux the video (no audio)
    subprocess.run(
        ["ffmpeg", "-y", "-i", silent, "-c:v", "copy", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not has_audio:
        print("   (source has no audio track — skipped)")


# ─── Main Effect ─────────────────────────────────────────────────────────────

def create_zoom_follow_effect(
    input_path, output_path,
    zoom_max=1.5, t_start=0, t_end=5,
    face_side="right", overlay_config=None,
    text_config=None, fade_mode="band",
):
    if overlay_config is None and text_config is not None:
        overlay_config = text_config

    print("1. Analyzing face trajectory …")
    raw_data, fps, (w, h) = get_face_data(input_path)
    face_data = smooth_data(raw_data, alpha=0.05)
    n_frames = len(face_data)

    # Overlay prep
    overlay = None
    if overlay_config:
        if overlay_config.get('type', 'text') == 'text':
            mfw = float(np.median(face_data[:, 2]))
            sfw = mfw * zoom_max
            mg = overlay_config.get('margin', 1.8)
            pad = int(w * 0.03)
            fcx = w * (0.72 if face_side == "right" else 0.28)
            pos = overlay_config.get('position', 'left')
            if pos == 'left':    aw = int(fcx - (sfw / 2 * mg) - pad)
            elif pos == 'right': aw = int(w - (fcx + sfw / 2 * mg) - pad)
            else:                aw = int(w * 0.5)
            overlay_config = {**overlay_config, '_avail_w': max(aw, 100), '_avail_h': int(h * 0.6)}
        overlay = create_overlay(overlay_config)

    # Timeline
    times = np.arange(n_frames) / fps
    p_raw = np.clip((times - t_start) / max(t_end - t_start, 1e-9), 0, 1)
    p_smooth = (p_raw * p_raw * (3.0 - 2.0 * p_raw)).astype(np.float32)
    zooms = (1.0 + (zoom_max - 1.0) * p_smooth).astype(np.float32)

    # Pre-allocate
    buf_warped = np.empty((h, w, 3), dtype=np.uint8)
    buf_out = np.empty((h, w, 3), dtype=np.uint8)
    buf_warped_f32 = np.empty((h, w, 3), dtype=np.float32)
    buf_fade_alpha = np.empty((h, w, 1), dtype=np.float32)
    buf_blend = np.empty((h, w, 3), dtype=np.float32)
    fade_bg_buf = np.empty((h, w, 3), dtype=np.float32)
    buf_rgb = np.empty((h, w, 3), dtype=np.uint8)
    dest_x_full = w * (0.28 if face_side == "left" else 0.72)

    # Hoist config lookups
    ovl_start = (overlay_config.get('t_start', t_start) if overlay_config else t_start)
    ovl_end = (overlay_config.get('t_end', t_end) if overlay_config else t_end)
    ovl_dur = max(ovl_end - ovl_start, 1e-9)
    ovl_pos = (overlay_config.get('position', 'left') if overlay_config else 'left')
    ovl_mg = (overlay_config.get('margin', 1.8) if overlay_config else 1.8)

    # ── Gradient fade setup ──────────────────────────────────────────
    edge_strip = max(int(w * EDGE_STRIP_FRAC), 1)
    fade_width = int(w * FADE_WIDTH_FRAC)
    ramp = np.linspace(0, 1, fade_width).astype(np.float32)
    base_gradient = np.ones((h, w), dtype=np.float32)
    if face_side == "right":
        base_gradient[:, :fade_width] = ramp[np.newaxis, :]
    else:
        base_gradient[:, w - fade_width:] = ramp[::-1][np.newaxis, :]
    base_gradient_3ch = base_gradient[:, :, np.newaxis]  # (h, w, 1) float32

    # ── Static lock state ────────────────────────────────────────────
    locked = False
    locked_M = None
    locked_fade_bg = None
    locked_sfx = locked_sfy = locked_sfw = locked_sfh = 0.0

    # I/O
    enc = detect_best_encoder()
    tmp = output_path + ".tmp_silent.mp4"
    writer = open_ffmpeg_writer(tmp, w, h, fps, enc)
    reader = ThreadedVideoReader(input_path, queue_size=64)

    print("2. Processing frames …")
    t0 = time.monotonic()

    for idx in range(n_frames):
        ok, bgr = reader.read()
        if not ok:
            break

        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=buf_rgb)
        rgb = buf_rgb
        t = times[idx]
        p = p_smooth[idx]
        z = zooms[idx]

        if locked:
            M = locked_M
            sfx, sfy = locked_sfx, locked_sfy
            sfw, sfh = locked_sfw, locked_sfh
        else:
            fx, fy, fw, fh = face_data[idx]
            tx = lerp(w / 2, float(fx), p)
            ty = lerp(h / 2, float(fy), p)
            dx = lerp(w / 2, dest_x_full, p)
            sx = dx - tx * z
            sy = h / 2 - ty * z

            M = np.float32([[z, 0, sx], [0, z, sy]])

            sfx = fx * z + sx
            sfy = fy * z + sy
            sfw = fw * z
            sfh = fh * z

            if p >= 1.0:
                locked = True
                locked_M = M
                locked_sfx, locked_sfy = sfx, sfy
                locked_sfw, locked_sfh = sfw, sfh

        # ── Warp with BORDER_REPLICATE (no black void) ───────────────
        cv2.warpAffine(rgb, M, (w, h), dst=buf_warped,
                       borderMode=cv2.BORDER_REPLICATE)

        # ── Gradient fade (per-row edge color band) ──────────────────
        if locked and locked_fade_bg is not None:
            fade_bg = locked_fade_bg
        else:
            # Sample thin edge strip, get per-row color
            if face_side == "right":
                edge_band = buf_warped[:, :edge_strip].mean(axis=1, dtype=np.float32)
            else:
                edge_band = buf_warped[:, w - edge_strip:].mean(axis=1, dtype=np.float32)

            # Downsample/upsample crush to destroy texture detail
            CRUSH_H = 6
            edge_col = edge_band.reshape(h, 1, 3)  # already float32 from .mean(dtype=)
            crushed = cv2.resize(edge_col, (1, CRUSH_H), interpolation=cv2.INTER_AREA)
            edge_band = cv2.resize(crushed, (1, h), interpolation=cv2.INTER_LINEAR).reshape(h, 3)

            fade_bg_buf[:] = edge_band[:, np.newaxis, :]  # tile in-place, stays float32
            fade_bg = fade_bg_buf

            if locked:
                locked_fade_bg = fade_bg.copy()

        # Blend: fade_alpha=1 → pure content, fade_alpha=0 → pure fade_bg
        buf_warped_f32[:] = buf_warped                          # uint8→float32 in-place
        np.multiply(base_gradient_3ch, p, out=buf_fade_alpha)   # p * gradient
        buf_fade_alpha += (1.0 - p)                             # (1-p) + p*gradient
        np.multiply(buf_warped_f32, buf_fade_alpha, out=buf_blend)
        np.subtract(1.0, buf_fade_alpha, out=buf_fade_alpha)
        np.multiply(fade_bg, buf_fade_alpha, out=buf_warped_f32)
        np.add(buf_blend, buf_warped_f32, out=buf_blend)
        np.clip(buf_blend, 0, 255, out=buf_blend)
        np.copyto(buf_out, buf_blend.astype(np.uint8))

        # ── Overlay ──────────────────────────────────────────────────
        if overlay and overlay_config and ovl_start <= t <= ovl_end:
            opacity = min((t - ovl_start) / ovl_dur * 4.0, 1.0)
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

    print("3. Muxing audio …")
    mux_audio(input_path, tmp, output_path)
    os.remove(tmp)
    print(f"Done → {output_path}")


# ─── Usage ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = int(time.time())

    create_zoom_follow_effect(
        input_path="longvid.mp4",
        output_path=f"longvid_band_{ts}.mp4",
        zoom_max=1.6, t_start=1.0, t_end=12.0,
        face_side="right",
        overlay_config={
            "content": "Band Mode", "position": "left",
            "color": "yellow", "t_start": 3.0, "t_end": 12.0,
            "margin": 1.8,
            "fontsize": 100,
        },
        fade_mode="band",
    )