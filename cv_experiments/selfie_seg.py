"""
Selfie Segmentation — Background Replacement Pipeline
======================================================
Uses MediaPipe's selfie segmentation to separate person from background,
then composites onto a new background. Three modes: portrait blur,
solid color, or centered text.
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
from moviepy.editor import TextClip


def lerp(a, b, t):
    return a + (b - a) * t


# ─── Encoder detection ───────────────────────────────────────────────────────


def _probe_encoder(name):
    try:
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=black:s=64x64:d=0.04",
                "-c:v",
                name,
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
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


# ─── Text background rendering ───────────────────────────────────────────────


class TextOverlay:
    def __init__(
        self,
        content,
        color="white",
        fontsize=80,
        font="Arial-Bold",
        max_width=None,
        max_height=None,
    ):
        fontsize = self._fit(content, fontsize, color, font, max_width, max_height)
        kw = dict(fontsize=fontsize, color=color, font=font)
        if max_width:
            kw.update(size=(max_width, None), method="caption")
        txt = TextClip(content, **kw)
        self.img = np.ascontiguousarray(txt.get_frame(0), dtype=np.float32)
        if txt.mask:
            m = txt.mask.get_frame(0)
            if m.max() > 1.0:
                m = m / 255.0
            self.mask = np.ascontiguousarray(m[:, :, np.newaxis], dtype=np.float32)
        else:
            self.mask = np.ones((*self.img.shape[:2], 1), dtype=np.float32)

    @staticmethod
    def _fit(content, fs, color, font, mw, mh):
        if not mw and not mh:
            return fs
        for s in range(fs, 19, -4):
            kw = dict(fontsize=s, color=color, font=font)
            if mw:
                kw.update(size=(mw, None), method="caption")
            sh = TextClip(content, **kw).get_frame(0).shape[:2]
            if (not mw or sh[1] <= mw) and (not mh or sh[0] <= mh):
                return s
        return 20


# ─── FFmpeg writer ───────────────────────────────────────────────────────────


def open_ffmpeg_writer(path, w, h, fps, enc):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        enc,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if enc == "libx264":
        cmd += ["-preset", "fast", "-crf", "18"]
    elif enc == "h264_nvenc":
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "20"]
    elif enc == "h264_videotoolbox":
        cmd += ["-q:v", "65"]
    cmd.append(path)
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def mux_audio(src, silent, out):
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            src,
        ],
        capture_output=True,
        text=True,
    )
    has_audio = probe.returncode == 0 and probe.stdout.strip() != ""

    if has_audio:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                silent,
                "-i",
                src,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                out,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        print(f"   Warning: Mux with audio failed (exit {result.returncode}):")
        print(f"     {result.stderr[-300:]}")
        print("     Falling back to video-only ...")

    subprocess.run(
        ["ffmpeg", "-y", "-i", silent, "-c:v", "copy", out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not has_audio:
        print("   (source has no audio track — skipped)")


# ─── Model helper ────────────────────────────────────────────────────────────


def _ensure_seg_model():
    """Return path to selfie segmentation model.
    Prefers the square 256x256 model (better quality than 144x256 landscape).
    Check local override first, then fall back to bundled."""
    # Prefer square model (256x256, higher quality)
    local_square = os.path.join(os.path.dirname(__file__), "selfie_segmenter.tflite")
    if os.path.isfile(local_square):
        return local_square
    # Fall back to landscape model
    local_landscape = os.path.join(
        os.path.dirname(__file__), "selfie_segmentation_landscape.tflite"
    )
    if os.path.isfile(local_landscape):
        return local_landscape
    # Bundled model inside the mediapipe package
    mp_dir = os.path.dirname(mp.__file__)
    bundled = os.path.join(
        mp_dir, "modules", "selfie_segmentation", "selfie_segmentation_landscape.tflite"
    )
    if os.path.isfile(bundled):
        return bundled
    raise FileNotFoundError(
        "Selfie segmentation model not found.\n"
        f"  Checked: {local_square}\n"
        f"  Checked: {local_landscape}\n"
        f"  Checked: {bundled}\n"
        "Download from https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter/float16/latest/"
        "selfie_segmenter.tflite"
    )


# ─── Mask dilation kernel ─────────────────────────────────────────────────────

def _dilate_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


# ─── Main Effect ─────────────────────────────────────────────────────────────


def apply_selfie_segmentation(
    input_path,
    output_path,
    mode="blur",
    bg_color=(255, 255, 255),
    blur_ksize=31,
    blur_sigma=15.0,
    feather_ksize=21,
    feather_sigma=11.0,
    mask_dilate=5,
    mask_threshold=0.5,
    mask_ema=0.3,
    skip_threshold=2.0,
    process_scale=0.5,
    text_content="hello",
    text_color="white",
    text_bg_color="black",
    text_fontsize=80,
    text_font="Arial-Bold",
):
    """
    Selfie segmentation with background replacement.

    Modes:
        "blur"  — portrait-style blurred background
        "color" — solid color background
        "text"  — centered text on solid background
    """
    assert mode in ("blur", "color", "text"), f"Unknown mode: {mode!r}"

    # ── Video info ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"   Input: {w}x{h} @ {fps:.1f} fps, ~{n_frames} frames")
    print(f"   Mode: {mode}")

    # ── Segmenter setup ──────────────────────────────────────────────
    model_path = _ensure_seg_model()
    print(f"   Model: {model_path}")

    opts = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    segmenter = vision.ImageSegmenter.create_from_options(opts)

    # ── Static backgrounds ───────────────────────────────────────────
    static_bg = None
    if mode == "color":
        static_bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
    elif mode == "text":
        # Render text and center on solid background
        overlay = TextOverlay(
            text_content, color=text_color, fontsize=text_fontsize, font=text_font
        )
        txt_img = overlay.img  # float32 RGB
        txt_mask = overlay.mask  # float32 (h_t, w_t, 1)
        th, tw = txt_img.shape[:2]

        # Parse text_bg_color
        if isinstance(text_bg_color, str):
            color_map = {
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "gray": (128, 128, 128),
                "grey": (128, 128, 128),
            }
            bg_rgb = color_map.get(text_bg_color.lower(), (0, 0, 0))
        else:
            bg_rgb = text_bg_color

        bg_frame = np.full((h, w, 3), bg_rgb, dtype=np.float32)
        # Center the text
        y0 = max(0, (h - th) // 2)
        x0 = max(0, (w - tw) // 2)
        y1 = min(h, y0 + th)
        x1 = min(w, x0 + tw)
        sy = 0 if y0 >= 0 else -y0
        sx = 0 if x0 >= 0 else -x0
        roi = bg_frame[y0:y1, x0:x1]
        t_slice = txt_img[sy : sy + y1 - y0, sx : sx + x1 - x0]
        m_slice = txt_mask[sy : sy + y1 - y0, sx : sx + x1 - x0]
        bg_frame[y0:y1, x0:x1] = t_slice * m_slice + roi * (1.0 - m_slice)
        static_bg = np.clip(bg_frame, 0, 255).astype(np.uint8)

    # ── Internal processing resolution ──────────────────────────────
    # Blur + composite at reduced res, upscale final output.
    # Segmenter already downscales to 256x256 internally.
    ps = process_scale
    pw, ph = int(w * ps), int(h * ps)
    sk = max(3, blur_ksize | 1)  # keep blur kernel odd
    fk = max(3, int(feather_ksize * ps) | 1)
    dk = max(1, int(mask_dilate * ps))
    print(f"   Process res: {pw}x{ph} (scale={ps})")

    # ── Pre-allocate buffers ────────────────────────────────────────
    # Segment + EMA at low-res, upscale mask, composite at full-res.
    # Only blur runs at low-res (that's the expensive part).
    buf_rgb = np.empty((h, w, 3), dtype=np.uint8)
    buf_small = np.empty((ph, pw, 3), dtype=np.uint8)
    buf_prev_small = np.zeros((ph, pw, 3), dtype=np.uint8)
    buf_mask = np.empty((ph, pw), dtype=np.float32)       # low-res mask
    buf_mask_full = np.empty((h, w), dtype=np.float32)     # upscaled mask
    buf_fg_f = np.empty((h, w, 3), dtype=np.float32)       # full-res
    buf_bg_f = np.empty((h, w, 3), dtype=np.float32)       # full-res
    buf_blend = np.empty((h, w, 3), dtype=np.float32)      # full-res
    buf_out = np.empty((h, w, 3), dtype=np.uint8)
    # Blur at low-res, then upscale
    buf_blur_small = np.empty((ph, pw, 3), dtype=np.uint8) if mode == "blur" else None
    buf_blur_full = np.empty((h, w, 3), dtype=np.uint8) if mode == "blur" else None
    # Static bg at full resolution
    if static_bg is not None:
        static_bg_f = static_bg.astype(np.float32)
    else:
        static_bg_f = None
    # EMA buffers at low resolution
    buf_mask_ema = np.zeros((ph, pw), dtype=np.float32)
    buf_ema_alpha = np.empty((ph, pw), dtype=np.float32)
    ema_initialized = False
    ema_attack = np.float32(min(mask_ema * 2.5, 1.0))
    ema_decay = np.float32(mask_ema * 0.5)

    # ── I/O ──────────────────────────────────────────────────────────
    enc = detect_best_encoder()
    tmp = output_path + ".tmp_silent.mp4"
    writer = open_ffmpeg_writer(tmp, w, h, fps, enc)
    reader = ThreadedVideoReader(input_path, queue_size=64)

    print(f"   Processing (mask_ema={mask_ema:.2f}, skip_thresh={skip_threshold}) ...")
    t0 = time.monotonic()
    last_ts = -1
    skipped = 0

    for idx in range(n_frames):
        ok, bgr = reader.read()
        if not ok:
            break

        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=buf_rgb)
        cv2.resize(buf_rgb, (pw, ph), dst=buf_small)

        # Monotonically increasing timestamp (ms)
        ts = int(idx * 1000 / fps) + 1
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts

        # Skip segmentation if frame barely changed
        frame_delta = float(cv2.absdiff(buf_small, buf_prev_small).mean())
        np.copyto(buf_prev_small, buf_small)

        if frame_delta > skip_threshold or not ema_initialized:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=buf_small)
            result = segmenter.segment_for_video(mp_img, ts)
            buf_mask[:] = result.confidence_masks[0].numpy_view()
            # Hard threshold → dilate → feather: clean edge, no halo
            cv2.threshold(buf_mask, mask_threshold, 1.0,
                          cv2.THRESH_BINARY, dst=buf_mask)
            if dk > 0:
                cv2.dilate(buf_mask, _dilate_kernel(dk), dst=buf_mask)
            cv2.GaussianBlur(buf_mask, (fk, fk), feather_sigma * ps, dst=buf_mask)
        else:
            # Advance segmenter state but reuse previous mask
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=buf_small)
            segmenter.segment_for_video(mp_img, ts)
            skipped += 1

        # Motion-adaptive asymmetric EMA
        if not ema_initialized:
            np.copyto(buf_mask_ema, buf_mask)
            ema_initialized = True
        else:
            delta = float(np.mean(np.abs(buf_mask - buf_mask_ema)))
            motion_boost = min(delta * 8.0, 1.0)
            np.copyto(buf_ema_alpha, np.where(buf_mask > buf_mask_ema, ema_attack, ema_decay))
            buf_ema_alpha += (1.0 - buf_ema_alpha) * motion_boost
            buf_mask_ema += buf_ema_alpha * (buf_mask - buf_mask_ema)
        # Upscale mask to full resolution (smooth edge at native res)
        cv2.resize(buf_mask_ema, (w, h), dst=buf_mask_full,
                   interpolation=cv2.INTER_LINEAR)
        mask_3ch = buf_mask_full[:, :, np.newaxis]

        # Composite at full resolution — sharp edges, no upscale halo
        buf_fg_f[:] = buf_rgb
        if mode == "blur":
            cv2.GaussianBlur(buf_small, (sk, sk), blur_sigma, dst=buf_blur_small)
            cv2.resize(buf_blur_small, (w, h), dst=buf_blur_full,
                       interpolation=cv2.INTER_LINEAR)
            buf_bg_f[:] = buf_blur_full
        else:
            buf_bg_f[:] = static_bg_f

        np.multiply(buf_fg_f, mask_3ch, out=buf_blend)
        inv_mask = 1.0 - mask_3ch
        np.multiply(buf_bg_f, inv_mask, out=buf_fg_f)
        np.add(buf_blend, buf_fg_f, out=buf_blend)
        np.copyto(buf_out, buf_blend.astype(np.uint8))

        writer.stdin.write(buf_out.tobytes())
        if idx % 50 == 0:
            print(f"   frame {idx}/{n_frames} (skipped {skipped})", flush=True)

    elapsed = time.monotonic() - t0
    actual = min(idx + 1, n_frames)
    print(
        f"   {actual} frames in {elapsed:.1f}s ({actual / max(elapsed, 0.01):.1f} fps)"
        f" — {skipped} skipped ({skipped * 100 // max(actual, 1)}%)"
    )

    reader.release()
    writer.stdin.close()
    writer.wait()
    segmenter.close()

    print("   Muxing audio ...")
    mux_audio(input_path, tmp, output_path)
    os.remove(tmp)
    print(f"Done -> {output_path}")


# ─── Realtime camera mode ─────────────────────────────────────────────────────


def realtime_selfie_segmentation(
    mode="blur",
    bg_color=(255, 255, 255),
    blur_ksize=31,
    blur_sigma=19.0,
    feather_ksize=21,
    feather_sigma=11.0,
    mask_dilate=5,
    mask_threshold=0.5,
    mask_ema=0.4,
    camera_index=0,
):
    """
    Live camera selfie segmentation with FPS overlay.
    Press 'q' to quit, 'm' to cycle modes (blur → color → back).

    Optimised for low latency:
    - Works in BGR throughout; only converts to RGB for MediaPipe
    - Zero per-frame heap allocations (all buffers pre-allocated)
    - In-place numpy ops with broadcast mask view (no 3-channel expand)
    - deque for O(1) FPS bookkeeping
    """
    from collections import deque

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    # Request 30fps — macOS cameras default to 15fps otherwise
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   Camera: {w}x{h} @ {actual_fps:.0f}fps")

    model_path = _ensure_seg_model()
    print(f"   Model: {model_path}")

    opts = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    segmenter = vision.ImageSegmenter.create_from_options(opts)

    # Pre-allocate everything
    buf_rgb = np.empty((h, w, 3), dtype=np.uint8)
    buf_mask = np.empty((h, w), dtype=np.float32)
    buf_mask_ema = np.zeros((h, w), dtype=np.float32)
    buf_ema_alpha = np.empty((h, w), dtype=np.float32)
    ema_initialized = False
    ema_attack = np.float32(mask_ema * 2.5)
    if ema_attack > 1.0:
        ema_attack = np.float32(1.0)
    ema_decay = np.float32(mask_ema * 0.5)
    buf_fg_f = np.empty((h, w, 3), dtype=np.float32)
    buf_bg_f = np.empty((h, w, 3), dtype=np.float32)
    buf_blend = np.empty((h, w, 3), dtype=np.float32)
    buf_out = np.empty((h, w, 3), dtype=np.uint8)
    buf_blur = np.empty((h, w, 3), dtype=np.uint8)
    # Static bg as BGR uint8 + float32 (pre-converted once)
    static_color_bg_bgr = np.full((h, w, 3), bg_color[::-1], dtype=np.uint8)
    static_color_bg_f = static_color_bg_bgr.astype(np.float32)

    modes = ["blur", "color"]
    mode_idx = modes.index(mode) if mode in modes else 0
    current_mode = modes[mode_idx]

    last_ts = -1
    frame_times = deque(maxlen=60)
    window_name = "Selfie Seg — press q to quit, m to cycle mode"

    print(f"   Mode: {current_mode} (press 'm' to cycle)")
    print("   Press 'q' to quit")

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        # MediaPipe needs RGB
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=buf_rgb)

        # Timestamp
        ts_ms = int(time.monotonic() * 1000)
        if ts_ms <= last_ts:
            ts_ms = last_ts + 1
        last_ts = ts_ms

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=buf_rgb)
        result = segmenter.segment_for_video(mp_img, ts_ms)
        buf_mask[:] = result.confidence_masks[0].numpy_view()

        # Hard threshold → dilate → feather: clean edge, no halo
        cv2.threshold(buf_mask, mask_threshold, 1.0,
                      cv2.THRESH_BINARY, dst=buf_mask)
        if mask_dilate > 0:
            cv2.dilate(buf_mask, _dilate_kernel(mask_dilate), dst=buf_mask)
        cv2.GaussianBlur(
            buf_mask, (feather_ksize, feather_ksize), feather_sigma, dst=buf_mask
        )

        # Motion-adaptive asymmetric EMA
        if not ema_initialized:
            np.copyto(buf_mask_ema, buf_mask)
            ema_initialized = True
        else:
            delta = float(np.mean(np.abs(buf_mask - buf_mask_ema)))
            motion_boost = min(delta * 8.0, 1.0)
            np.copyto(buf_ema_alpha, np.where(buf_mask > buf_mask_ema, ema_attack, ema_decay))
            buf_ema_alpha += (1.0 - buf_ema_alpha) * motion_boost
            buf_mask_ema += buf_ema_alpha * (buf_mask - buf_mask_ema)
        np.copyto(buf_mask, buf_mask_ema)

        # Composite in BGR space — zero allocs
        buf_fg_f[:] = bgr  # work in BGR, skip back-convert
        if current_mode == "blur":
            cv2.GaussianBlur(bgr, (blur_ksize, blur_ksize), blur_sigma, dst=buf_blur)
            buf_bg_f[:] = buf_blur
        else:
            buf_bg_f[:] = static_color_bg_f

        mask_3ch = buf_mask[:, :, np.newaxis]  # (h, w, 1) view, no copy
        np.multiply(buf_fg_f, mask_3ch, out=buf_blend)
        # Reuse buf_fg_f as temp for bg contribution
        np.subtract(1.0, mask_3ch, out=mask_3ch)
        np.multiply(buf_bg_f, mask_3ch, out=buf_fg_f)
        np.add(buf_blend, buf_fg_f, out=buf_blend)
        np.copyto(buf_out, buf_blend.astype(np.uint8))

        # FPS (O(1) deque)
        now = time.monotonic()
        frame_times.append(now)
        if len(frame_times) >= 2:
            fps_val = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        else:
            fps_val = 0.0

        # Draw FPS + mode
        label = f"{fps_val:.1f} fps | mode: {current_mode}"
        cv2.putText(
            buf_out,
            label,
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            buf_out,
            label,
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, buf_out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            mode_idx = (mode_idx + 1) % len(modes)
            current_mode = modes[mode_idx]
            print(f"   Switched to: {current_mode}")

    cap.release()
    segmenter.close()
    cv2.destroyAllWindows()
    print("Done.")


# ─── Usage ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--realtime" in sys.argv:
        realtime_selfie_segmentation(mode="blur")
        sys.exit(0)

    ts = int(time.time())

    # # 1. Portrait blur mode
    # apply_selfie_segmentation(
    #     input_path="longvid.mp4",
    #     output_path=f"longvid_seg_blur_{ts}.mp4",
    #     mode="blur",
    #     blur_ksize=51,
    #     blur_sigma=25.0,
    # )

    # 2. Solid color mode
    apply_selfie_segmentation(
        input_path="longvid.mp4",
        output_path=f"longvid_seg_color_{ts}.mp4",
        mode="color",
        bg_color=(30, 30, 30),
    )

    # # 3. Text mode
    # apply_selfie_segmentation(
    #     input_path="longvid.mp4",
    #     output_path=f"longvid_seg_text_{ts}.mp4",
    #     mode="text",
    #     text_content="hello world",
    #     text_color="white",
    #     text_bg_color="black",
    #     text_fontsize=120,
    # )
