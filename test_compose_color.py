"""Debug: does FFV1 bgr0 → libx264 (no pix_fmt) cause green tint?"""

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


def _read_frame(path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def main():
    tmpdir = tempfile.mkdtemp(prefix="compose_color_")
    print(f"Dir: {tmpdir}\n")

    ref_frame = _read_frame(INPUT)
    print(f"Reference from original HEVC: shape={ref_frame.shape}")

    try:
        for intermediate_fmt in ["bgr0", "yuv444p", "yuv420p"]:
            # Step 1: HEVC → FFV1 with given pix_fmt (simulate phase output)
            ffv1_path = os.path.join(tmpdir, f"intermediate_{intermediate_fmt}.avi")
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", INPUT, "-frames:v", "5",
                "-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "4", "-context", "0",
                "-pix_fmt", intermediate_fmt, "-an", ffv1_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # Step 2: FFV1 → H.264 (exactly as compose.py does it — NO -pix_fmt)
            h264_no_pix = os.path.join(tmpdir, f"compose_nopix_{intermediate_fmt}.mp4")
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-i", ffv1_path,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-an", h264_no_pix,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.stderr.strip():
                print(f"  [{intermediate_fmt} → h264 (no pix_fmt)] stderr: {r.stderr.strip()}")

            # Step 3: FFV1 → H.264 WITH explicit -pix_fmt yuv420p
            h264_with_pix = os.path.join(tmpdir, f"compose_withpix_{intermediate_fmt}.mp4")
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-i", ffv1_path,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-an", h264_with_pix,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.stderr.strip():
                print(f"  [{intermediate_fmt} → h264 (yuv420p)] stderr: {r.stderr.strip()}")

            # Compare both against reference
            frame_no_pix = _read_frame(h264_no_pix)
            frame_with_pix = _read_frame(h264_with_pix)

            if frame_no_pix is not None:
                diff_no = np.abs(ref_frame.astype(int) - frame_no_pix.astype(int))
                ch_no = np.mean(diff_no, axis=(0, 1))
                max_no = diff_no.max()
                # Check actual negotiated pix_fmt
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", h264_no_pix],
                    capture_output=True, text=True)
                negotiated = r.stdout.strip()
                status = "OK" if max_no < 10 else "BAD"
                print(f"  {intermediate_fmt:10s} → h264 (auto={negotiated:10s}): "
                      f"max={max_no:3d} mean_BGR=[{ch_no[0]:.1f},{ch_no[1]:.1f},{ch_no[2]:.1f}] [{status}]")

            if frame_with_pix is not None:
                diff_with = np.abs(ref_frame.astype(int) - frame_with_pix.astype(int))
                ch_with = np.mean(diff_with, axis=(0, 1))
                max_with = diff_with.max()
                status = "OK" if max_with < 10 else "BAD"
                print(f"  {intermediate_fmt:10s} → h264 (forced yuv420p): "
                      f"max={max_with:3d} mean_BGR=[{ch_with[0]:.1f},{ch_with[1]:.1f},{ch_with[2]:.1f}] [{status}]")
            print()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
