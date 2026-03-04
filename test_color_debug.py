"""Debug: compare pixel values between original and FFV1 bgr0 to find color shift."""

import subprocess
import sys
import os
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cv2

INPUT = "/Users/sidhant/sidhant-experiments/input.mp4"
W, H, FPS = 1920, 1080, 60

FORMATS_TO_TEST = ["bgr0", "bgra", "yuv444p", "yuv420p"]


def _extract_ffv1(input_path, output_path, pix_fmt):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path, "-frames:v", "1",
        "-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "4", "-context", "0",
        "-pix_fmt", pix_fmt, "-an", output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg failed for {pix_fmt}: {r.stderr}")
        return False
    return True


def _read_frame(path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def _simulate_active_pipeline(output_path, pix_fmt):
    """Simulate: OpenCV reads frame → pipes BGR24 → FFV1 with given pix_fmt."""
    cap = cv2.VideoCapture(INPUT)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "pipe:0",
         "-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "4", "-context", "0",
         "-pix_fmt", pix_fmt, "-an", output_path],
        stdin=subprocess.PIPE)
    try:
        proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass
    try:
        proc.stdin.close()
    except Exception:
        pass
    proc.wait()
    return proc.returncode == 0


def main():
    tmpdir = tempfile.mkdtemp(prefix="color_debug_")
    print(f"Dir: {tmpdir}\n")

    try:
        # Reference: read frame 0 from original HEVC
        ref_frame = _read_frame(INPUT)
        print(f"Reference frame shape: {ref_frame.shape}, dtype: {ref_frame.dtype}")
        # Sample some pixels
        sample_points = [(100, 100), (540, 960), (800, 1500), (200, 500)]
        print(f"Reference pixel values (BGR):")
        for y, x in sample_points:
            print(f"  ({y},{x}): {ref_frame[y, x]}")

        print()

        # Test each pix_fmt: transcode path (HEVC → FFV1)
        print("== TRANSCODE PATH (HEVC → FFV1 → OpenCV read) ==")
        for fmt in FORMATS_TO_TEST:
            path = os.path.join(tmpdir, f"transcode_{fmt}.avi")
            if not _extract_ffv1(INPUT, path, fmt):
                continue
            frame = _read_frame(path)
            if frame is None:
                print(f"  {fmt}: FAILED to read")
                continue

            diffs = []
            for y, x in sample_points:
                diff = ref_frame[y, x].astype(int) - frame[y, x].astype(int)
                diffs.append(diff)

            max_diff = max(abs(d).max() for d in diffs)
            mean_diff = np.mean([np.mean(np.abs(ref_frame.astype(int) - frame.astype(int)))])
            # Check channel-wise mean to detect green shift
            channel_means = np.mean(np.abs(ref_frame.astype(int) - frame.astype(int)), axis=(0, 1))

            status = "OK" if max_diff < 10 else "DRIFT" if max_diff < 30 else "BAD"
            print(f"  {fmt:12s}: max_pixel_diff={max_diff:3d}, mean_diff={mean_diff:.1f}, "
                  f"per_channel(B,G,R)=[{channel_means[0]:.1f}, {channel_means[1]:.1f}, {channel_means[2]:.1f}] [{status}]")

        print()

        # Test each pix_fmt: active path (OpenCV → BGR24 pipe → FFV1 → OpenCV read)
        print("== ACTIVE PATH (OpenCV → pipe BGR24 → FFV1 → OpenCV read) ==")
        for fmt in FORMATS_TO_TEST:
            path = os.path.join(tmpdir, f"active_{fmt}.avi")
            if not _simulate_active_pipeline(path, fmt):
                print(f"  {fmt}: FAILED to create")
                continue
            frame = _read_frame(path)
            if frame is None:
                print(f"  {fmt}: FAILED to read")
                continue

            max_diff = np.max(np.abs(ref_frame.astype(int) - frame.astype(int)))
            mean_diff = np.mean(np.abs(ref_frame.astype(int) - frame.astype(int)))
            channel_means = np.mean(np.abs(ref_frame.astype(int) - frame.astype(int)), axis=(0, 1))

            status = "OK" if max_diff < 10 else "DRIFT" if max_diff < 30 else "BAD"
            print(f"  {fmt:12s}: max_pixel_diff={max_diff:3d}, mean_diff={mean_diff:.1f}, "
                  f"per_channel(B,G,R)=[{channel_means[0]:.1f}, {channel_means[1]:.1f}, {channel_means[2]:.1f}] [{status}]")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
