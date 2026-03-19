"""Remotion Studio process management and preview asset writing."""

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

from video_effects.helpers.remotion import _get_remotion_dir

logger = logging.getLogger(__name__)


def _preview_dir() -> Path:
    return _get_remotion_dir() / "public" / "_preview"


def _link_or_copy(src: str, dst: Path) -> None:
    """Hard-link src to dst, falling back to copy if cross-device."""
    if dst.exists() and os.path.samefile(src, dst):
        return
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_preview_assets(
    mg_plan: dict,
    base_video_path: str,
    face_data_path: str,
    zoom_state_path: str,
    video_info: dict,
) -> None:
    """Create _preview/ with hard-linked assets and plan JSON.

    Hard links (not symlinks) so Remotion's webpack bundler can copy them
    into its temp bundle directory.
    """
    preview = _preview_dir()
    preview.mkdir(parents=True, exist_ok=True)

    # Link base video
    if base_video_path and os.path.exists(base_video_path):
        _link_or_copy(base_video_path, preview / "base.mp4")

    # Write plan JSON with video dimensions
    plan = {
        **mg_plan,
        "durationInFrames": int(video_info.get("total_frames", 300)),
        "fps": int(video_info.get("fps", 30)),
        "width": int(video_info.get("width", 1920)),
        "height": int(video_info.get("height", 1080)),
    }
    (preview / "plan.json").write_text(json.dumps(plan, indent=2))

    # Link face data
    if face_data_path and os.path.exists(face_data_path):
        _link_or_copy(face_data_path, preview / "face_data.json")

    # Link zoom state
    if zoom_state_path and os.path.exists(zoom_state_path):
        _link_or_copy(zoom_state_path, preview / "zoom_state.json")


def cleanup_preview() -> None:
    """Remove the _preview/ directory."""
    preview = _preview_dir()
    if preview.exists():
        shutil.rmtree(preview, ignore_errors=True)


def update_preview_plan(mg_plan: dict, video_info: dict) -> None:
    """Rewrite only plan.json for hot-reload on rejection."""
    preview = _preview_dir()
    plan = {
        **mg_plan,
        "durationInFrames": int(video_info.get("total_frames", 300)),
        "fps": int(video_info.get("fps", 30)),
        "width": int(video_info.get("width", 1920)),
        "height": int(video_info.get("height", 1080)),
    }
    (preview / "plan.json").write_text(json.dumps(plan, indent=2))


def _find_free_port(start: int, end: int) -> int | None:
    """Return first free port in range, or None."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return None


def _kill_orphan(pid_file: Path) -> None:
    """Kill a stale Studio process from a previous run."""
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        logger.info("Killed orphan Studio PID %d", pid)
    except (ProcessLookupError, PermissionError, ValueError, OSError):
        pass
    pid_file.unlink(missing_ok=True)


def start_studio(port: int = 3100) -> dict:
    """Start Remotion Studio, return {studio_url, pid, port}."""
    preview = _preview_dir()
    preview.mkdir(parents=True, exist_ok=True)
    pid_file = preview / ".studio_pid"

    _kill_orphan(pid_file)

    free_port = _find_free_port(port, port + 10)
    if free_port is None:
        logger.warning("No free port in %d-%d, skipping Studio", port, port + 10)
        return {"studio_url": "", "pid": 0, "port": 0}

    remotion_dir = _get_remotion_dir()
    proc = subprocess.Popen(
        ["npx", "remotion", "studio", "--port", str(free_port)],
        cwd=str(remotion_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    pid_file.write_text(str(proc.pid))

    # Poll for readiness
    url = f"http://localhost:{free_port}"
    for _ in range(30):
        time.sleep(0.5)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", free_port)) == 0:
                    logger.info("Remotion Studio ready at %s (PID %d)", url, proc.pid)
                    return {"studio_url": url, "pid": proc.pid, "port": free_port}
        except OSError:
            continue

    logger.warning("Studio did not become ready in 15s")
    return {"studio_url": url, "pid": proc.pid, "port": free_port}


def stop_studio(pid: int) -> None:
    """Stop Studio process and clean up preview dir."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
        else:
            os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass

    # Clean up preview directory
    preview = _preview_dir()
    if preview.exists():
        shutil.rmtree(preview, ignore_errors=True)
