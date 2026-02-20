"""
Zoom-Bounce Nuclear — Zero-Copy FFmpeg Architecture
=====================================================
Shifts Python's role from pixel executor to math compiler.
All rendering is delegated to FFmpeg's optimized C filtergraphs
via dynamically compiled expressions.

Three phases:
1. Analysis  — MediaPipe face detection (reused from zoom_bounce.py)
2. Compilation — Python generates FFmpeg filter expressions
3. Execution — FFmpeg subprocess renders each segment; no video bytes in Python
"""

import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from zoom_bounce import (
    _parse_events,
    build_bounce_curves,
    build_effect_curves,
    get_face_data_seek,
    get_face_data,
    smooth_data,
    _compute_active_frame_ranges,
    _compute_render_ranges,
    _probe_source_codec,
    _probe_keyframe_times,
    _extract_passthrough,
    _concat_segments,
    mux_audio,
    detect_best_encoder,
    _run_ffmpeg_with_progress,
    _render_hold_ffmpeg,
    EASE_FUNCTIONS,
    EASE_IN_FUNCTIONS,
    EASE_OUT_FUNCTIONS,
)

# ─── Step 2: Easing Expression Compilers ─────────────────────────────────────
# Each maps an easing name to a function: t_expr -> FFmpeg C-expression string

def _ease_smooth(T):
    return f"sin(PI*{T})"

def _ease_snap(T):
    return (f"if(lt({T},0.25),pow({T}/0.25\\,2),"
            f"if(gt({T},0.75),1-pow(({T}-0.75)/0.25\\,2),1))")

def _ease_overshoot(T):
    return f"clip(sin(PI*{T})+0.15*sin(2*PI*{T}),0,1.15)"

def _ease_in_smooth(T):
    return f"sin(PI/2*{T})"

def _ease_in_snap(T):
    return f"if(lt({T},0.5),pow({T}/0.5\\,2),1)"

def _ease_in_overshoot(T):
    return f"clip(sin(PI/2*{T})+0.15*sin(PI*{T}),0,1.15)"

def _ease_out_smooth(T):
    return f"cos(PI/2*{T})"

def _ease_out_snap(T):
    return f"clip(1-pow({T}\\,2),0,1)"

def _ease_out_overshoot(T):
    return f"clip(cos(PI/2*{T})-0.15*sin(PI*{T}),0,1.15)"

FFMPEG_EASE = {
    "smooth": _ease_smooth,
    "snap": _ease_snap,
    "overshoot": _ease_overshoot,
}

FFMPEG_EASE_IN = {
    "smooth": _ease_in_smooth,
    "snap": _ease_in_snap,
    "overshoot": _ease_in_overshoot,
}

FFMPEG_EASE_OUT = {
    "smooth": _ease_out_smooth,
    "snap": _ease_out_snap,
    "overshoot": _ease_out_overshoot,
}


# ─── Step 3: Face Data Expression Compiler ───────────────────────────────────

def _build_lerp_chain(kf_times, kf_values, t_var="t"):
    """
    Build chained if(between(t,...), lerp(...), ...) FFmpeg expression
    from keyframe times and values.  Generalises the pattern from
    _render_hold_ffmpeg.
    """
    if len(kf_values) == 1:
        return f"{kf_values[0]:.1f}"
    # Build from last segment backwards
    expr = f"{kf_values[-1]:.1f}"
    for i in range(len(kf_values) - 2, -1, -1):
        t0 = kf_times[i]
        t1 = kf_times[i + 1]
        v0 = kf_values[i]
        v1 = kf_values[i + 1]
        seg_dur = t1 - t0
        if seg_dur <= 0:
            continue
        seg_expr = f"lerp({v0:.1f}\\,{v1:.1f}\\,({t_var}-{t0:.4f})/{seg_dur:.4f})"
        expr = f"if(between({t_var}\\,{t0:.4f}\\,{t1:.4f})\\,{seg_expr}\\,{expr})"
    return expr


def _sample_face_keyframes(face_data_stable, frame_start, frame_end, fps,
                           interval=0.5):
    """
    Sample face positions every `interval` seconds with +-1s averaging window.
    Returns (kf_times, kf_fx, kf_fy) — times relative to segment start.
    """
    n = frame_end - frame_start + 1
    avg_window = int(fps)  # +-1 second

    key_interval_frames = max(1, int(interval * fps))
    keyframe_indices = list(range(0, n, key_interval_frames))
    if keyframe_indices[-1] != n - 1:
        keyframe_indices.append(n - 1)

    kf_times = []
    kf_fx = []
    kf_fy = []

    for ki in keyframe_indices:
        lo = max(0, ki - avg_window)
        hi = min(n, ki + avg_window + 1)
        fx_sum = 0.0
        fy_sum = 0.0
        count = hi - lo
        for j in range(lo, hi):
            abs_idx = frame_start + j
            fx_sum += float(face_data_stable[abs_idx][0])
            fy_sum += float(face_data_stable[abs_idx][1])
        kf_times.append(ki / fps)
        kf_fx.append(fx_sum / count)
        kf_fy.append(fy_sum / count)

    return kf_times, kf_fx, kf_fy


def _compile_face_expressions(face_data_stable, frame_start, frame_end, fps,
                              t_var="t"):
    """
    Compile piecewise-linear FFmpeg expressions for face x/y position
    over the segment timespan.  Returns (fx_expr, fy_expr).
    """
    kf_times, kf_fx, kf_fy = _sample_face_keyframes(
        face_data_stable, frame_start, frame_end, fps
    )
    fx_expr = _build_lerp_chain(kf_times, kf_fx, t_var)
    fy_expr = _build_lerp_chain(kf_times, kf_fy, t_var)
    return fx_expr, fy_expr


# ─── Step 4: Zoom Geometry Compiler ──────────────────────────────────────────

def _compile_zoom_expr(event, fps, w, h, dest_x_full, fx_expr, fy_expr,
                       t_var="t", t_offset=0.0):
    """
    Compile FFmpeg crop expressions for a single event.

    The affine warp M = [[z,0,sx],[0,z,sy]] is equivalent to:
      crop_w = iw/z,  crop_h = ih/z
      crop_x = clip(tx - dx/z, 0, iw - crop_w)
      crop_y = clip(ty - ih/(2*z), 0, ih - crop_h)
    where tx = iw/2 + (fx - iw/2)*p, ty = ih/2 + (fy - ih/2)*p,
          dx = iw/2 + (dest_x - iw/2)*p.

    Returns: (crop_w_expr, crop_h_expr, crop_x_expr, crop_y_expr, scale_w, scale_h)
    """
    action = event["action"]
    bs = event["start"] - t_offset
    be = event["end"] - t_offset
    dur = max(be - bs, 1e-9)
    zm = event["zoom"]
    mode = event["ease"]

    # t_norm expression: normalized time within this event [0,1]
    t_norm_expr = f"clip(({t_var}-{bs:.4f})/{dur:.4f},0,1)"

    # Select easing function
    if action == "bounce":
        ease_fn = FFMPEG_EASE[mode]
    elif action == "in":
        ease_fn = FFMPEG_EASE_IN[mode]
    elif action == "out":
        ease_fn = FFMPEG_EASE_OUT[mode]
    else:
        return None

    # p (intensity) expression
    p_expr = ease_fn(t_norm_expr)

    # z (zoom) expression: 1 + (zm - 1) * p
    z_factor = zm - 1.0
    z_expr = f"(1+{z_factor:.4f}*({p_expr}))"

    # Geometry: crop from the original frame
    # tx = w/2 + (fx - w/2)*p => center + face_offset*p
    half_w = w / 2.0
    half_h = h / 2.0
    # dest_x offset from center
    dest_off = dest_x_full - half_w

    # crop dimensions
    cw_expr = f"({w}/{z_expr})"
    ch_expr = f"({h}/{z_expr})"

    # Pan target: tx = w/2 + (fx - w/2)*p
    tx_expr = f"({half_w}+({fx_expr}-{half_w})*({p_expr}))"
    ty_expr = f"({half_h}+({fy_expr}-{half_h})*({p_expr}))"

    # Destination x: dx = w/2 + (dest_x_full - w/2)*p
    dx_expr = f"({half_w}+{dest_off:.1f}*({p_expr}))"

    # crop_x = tx - dx/z  (where the visible center maps to)
    # Actually: from the affine: sx = dx - tx*z => crop_x = tx - dx/z
    # More precisely: the crop top-left in source coords:
    # crop_x = tx - (dx / z)  — this needs derivation from the warp
    # Let me re-derive: warp is out[x] = z*in[x] + sx, where sx = dx - tx*z
    # So in[x] = (out[x] - sx)/z.  Top-left of output (0,0) maps to:
    #   in_x = -sx/z = -(dx - tx*z)/z = tx - dx/z
    # That's crop_x.  crop_y similarly: sy = h/2 - ty*z
    #   in_y = -sy/z = -(h/2 - ty*z)/z = ty - h/(2*z)
    crop_x_expr = f"clip({tx_expr}-{dx_expr}/{z_expr},0,{w}-{cw_expr})"
    crop_y_expr = f"clip({ty_expr}-{half_h}/{z_expr},0,{h}-{ch_expr})"

    return cw_expr, ch_expr, crop_x_expr, crop_y_expr, w, h, p_expr, z_expr


def _compile_timeline_crop(events, fps, w, h, dest_x_full, fx_expr, fy_expr,
                           t_start, t_end, t_var="t"):
    """
    Chain events with if(between(t,...), ..., passthrough) for the full
    segment timeline.  Max-wins for overlapping events (higher zoom wins).

    Returns a single FFmpeg crop filter string or None if no zoom events.
    """
    zoom_events = [e for e in events if e["action"] in ("bounce", "in", "out")]
    if not zoom_events:
        return None

    t_offset = t_start

    # Build per-event crop expressions
    event_exprs = []
    for ev in zoom_events:
        result = _compile_zoom_expr(
            ev, fps, w, h, dest_x_full, fx_expr, fy_expr,
            t_var=t_var, t_offset=t_offset,
        )
        if result is None:
            continue
        cw, ch, cx, cy, sw, sh, p_expr, z_expr = result
        bs = ev["start"] - t_offset
        be = ev["end"] - t_offset
        event_exprs.append((bs, be, cw, ch, cx, cy, p_expr, z_expr))

    if not event_exprs:
        return None

    # For holds between in→out: p=1, z=in_zoom (constant)
    # These are identified by the build_bounce_curves logic already,
    # but we need to handle them in the expression chain too.
    # We handle this by building a unified p_expr and z_expr timeline.

    # Build chained crop_w, crop_h, crop_x, crop_y expressions
    # Passthrough = identity crop (full frame)
    pass_cw = str(w)
    pass_ch = str(h)
    pass_cx = "0"
    pass_cy = "0"

    # Chain from last event backwards — later events override
    cw_chain = pass_cw
    ch_chain = pass_ch
    cx_chain = pass_cx
    cy_chain = pass_cy

    for bs, be, cw, ch_e, cx, cy, p_expr, z_expr in reversed(event_exprs):
        cw_chain = f"if(between({t_var}\\,{bs:.4f}\\,{be:.4f})\\,{cw}\\,{cw_chain})"
        ch_chain = f"if(between({t_var}\\,{bs:.4f}\\,{be:.4f})\\,{ch_e}\\,{ch_chain})"
        cx_chain = f"if(between({t_var}\\,{bs:.4f}\\,{be:.4f})\\,{cx}\\,{cx_chain})"
        cy_chain = f"if(between({t_var}\\,{bs:.4f}\\,{be:.4f})\\,{cy}\\,{cy_chain})"

    # Handle hold regions (between in-end and out-start)
    hold_regions = _find_hold_regions(events, t_offset)
    for h_start, h_end, h_zoom in hold_regions:
        # During hold: constant zoom at h_zoom, p=1
        hold_cw = f"{w / h_zoom:.1f}"
        hold_ch = f"{h / h_zoom:.1f}"
        # Face position during hold — use the fx/fy expressions directly
        hold_cx = f"clip({fx_expr}-{dest_x_full}/{h_zoom:.4f},0,{w}-{hold_cw})"
        hold_cy = f"clip({fy_expr}-{h / 2.0}/{h_zoom:.4f},0,{h}-{hold_ch})"

        cw_chain = f"if(between({t_var}\\,{h_start:.4f}\\,{h_end:.4f})\\,{hold_cw}\\,{cw_chain})"
        ch_chain = f"if(between({t_var}\\,{h_start:.4f}\\,{h_end:.4f})\\,{hold_ch}\\,{ch_chain})"
        cx_chain = f"if(between({t_var}\\,{h_start:.4f}\\,{h_end:.4f})\\,{hold_cx}\\,{cx_chain})"
        cy_chain = f"if(between({t_var}\\,{h_start:.4f}\\,{h_end:.4f})\\,{hold_cy}\\,{cy_chain})"

    return cw_chain, ch_chain, cx_chain, cy_chain


def _find_hold_regions(events, t_offset):
    """Find hold regions between paired in→out events.  Returns [(start, end, zoom)]."""
    holds = []
    in_end = None
    in_zoom = None
    for ev in events:
        if ev["action"] == "in":
            in_end = ev["end"] - t_offset
            in_zoom = ev["zoom"]
        elif ev["action"] == "out" and in_end is not None:
            out_start = ev["start"] - t_offset
            if out_start > in_end:
                holds.append((in_end, out_start, in_zoom))
            in_end = None
            in_zoom = None
    return holds


# ─── Step 5: Effect Compilers (sendcmd-based) ────────────────────────────────
# FFmpeg's gblur/boxblur don't support per-frame 't' expressions in their
# parameters. We use sendcmd to step filter parameters at regular intervals.

MAX_BLUR_SIGMA = 20.0   # gblur sigma ceiling for zoom_blur
MAX_WHIP_RADIUS = 40    # boxblur radius ceiling for whip
SENDCMD_STEP = 1        # frames between sendcmd updates (1 = every frame)


def _compile_blur_sendcmd(events, fps, t_offset, step_frames=SENDCMD_STEP):
    """
    Generate sendcmd lines that step gblur@blur sigma per frame.
    Returns list of "time target command arg" strings, or None.
    """
    blur_events = [e for e in events if e["action"] == "zoom_blur"]
    if not blur_events:
        return None

    # Collect all frame times that fall within blur events
    commands = []
    for ev in blur_events:
        bs = ev["start"] - t_offset
        be = ev["end"] - t_offset
        dur = max(be - bs, 1e-9)
        intensity = ev.get("intensity", 1.0)
        # Sample at frame intervals
        frame_start = max(0, int(bs * fps))
        frame_end = int(be * fps) + 1
        for f in range(frame_start, frame_end + 1, step_frames):
            t = f / fps
            t_norm = max(0.0, min(1.0, (t - bs) / dur))
            strength = math.sin(math.pi * t_norm) * intensity
            sigma = strength * MAX_BLUR_SIGMA
            if sigma < 0.01:
                sigma = 0.0
            commands.append(f"{t:.4f} blur sigma {sigma:.2f}")

    return commands if commands else None


def _compile_whip_sendcmd(events, fps, t_offset, step_frames=SENDCMD_STEP):
    """
    Generate sendcmd lines that step boxblur@whip luma_radius per frame.
    Returns list of "time target command arg" strings, or None.
    """
    whip_events = [e for e in events if e["action"] == "whip"]
    if not whip_events:
        return None

    commands = []
    for ev in whip_events:
        bs = ev["start"] - t_offset
        be = ev["end"] - t_offset
        dur = max(be - bs, 1e-9)
        intensity = ev.get("intensity", 1.0)
        direction = ev.get("direction", "h")
        frame_start = max(0, int(bs * fps))
        frame_end = int(be * fps) + 1
        for f in range(frame_start, frame_end + 1, step_frames):
            t = f / fps
            t_norm = max(0.0, min(1.0, (t - bs) / dur))
            strength = math.sin(math.pi * t_norm) * intensity
            radius = int(strength * MAX_WHIP_RADIUS)
            radius = max(0, min(MAX_WHIP_RADIUS, radius))
            if direction == "h":
                commands.append(f"{t:.4f} whip luma_radius {radius}")
            else:
                # For vertical: use chroma_radius as proxy (luma = vertical)
                # Actually boxblur applies same radius in all directions.
                # We use separate h/v boxblur filters for each direction.
                commands.append(f"{t:.4f} whip luma_radius {radius}")
    return commands if commands else None


# ─── Step 6: Filtergraph Builder ─────────────────────────────────────────────

def _build_segment_filtergraph(events, face_data_stable, frame_start, frame_end,
                               fps, w, h, face_side, dest_x_full, tmp_dir,
                               seg_idx):
    """
    Build -vf filter string and optional sendcmd file for an active segment.

    Pipeline: [sendcmd if effects] -> crop(dynamic) -> scale(w:h:bilinear)
              -> [gblur@blur if blur] -> [boxblur@whip if whip]

    Returns: (filter_string, sendcmd_path_or_None)
    """
    t_start = frame_start / fps
    t_end = (frame_end + 1) / fps
    t_var = "t"
    t_offset = t_start

    # Compile face position expressions
    fx_expr, fy_expr = _compile_face_expressions(
        face_data_stable, frame_start, frame_end, fps, t_var=t_var,
    )

    # Filter events to those overlapping this segment
    seg_events = _events_in_range(events, t_start, t_end)

    # Compile zoom crop
    crop_result = _compile_timeline_crop(
        seg_events, fps, w, h, dest_x_full, fx_expr, fy_expr,
        t_start, t_end, t_var=t_var,
    )

    filters = []
    sendcmd_lines = []

    # Blur effect: compile sendcmd for gblur
    blur_cmds = _compile_blur_sendcmd(seg_events, fps, t_offset)
    has_blur = blur_cmds is not None

    # Whip effect: compile sendcmd for boxblur
    whip_cmds = _compile_whip_sendcmd(seg_events, fps, t_offset)
    has_whip = whip_cmds is not None

    # Collect all sendcmd lines
    if blur_cmds:
        sendcmd_lines.extend(blur_cmds)
    if whip_cmds:
        sendcmd_lines.extend(whip_cmds)

    # Write sendcmd file if needed
    sendcmd_path = None
    if sendcmd_lines:
        sendcmd_lines.sort(key=lambda s: float(s.split()[0]))
        sendcmd_path = os.path.join(tmp_dir, f"sendcmd_{seg_idx:04d}.txt")
        with open(sendcmd_path, "w") as f:
            for line in sendcmd_lines:
                f.write(line + ";\n")
        filters.append(f"sendcmd=f='{sendcmd_path}'")

    if crop_result:
        cw_expr, ch_expr, cx_expr, cy_expr = crop_result
        filters.append(f"crop='{cw_expr}':'{ch_expr}':'{cx_expr}':'{cy_expr}'")
        filters.append(f"scale={w}:{h}:flags=bilinear")

    if has_blur:
        filters.append("gblur@blur=sigma=0")

    if has_whip:
        filters.append("boxblur@whip=lr=0:lp=1")

    if not filters:
        filters.append("copy")

    return ",".join(filters), sendcmd_path


# ─── Step 7: Nuclear Segment Renderer ────────────────────────────────────────

def _render_active_segment_nuclear(input_path, output_path, frame_start, frame_end,
                                   events, face_data_stable, p_curve, zooms,
                                   blur_strength, whip_strength,
                                   fps, w, h, face_side, dest_x_full,
                                   enc, tmp_dir, seg_idx):
    """
    Render one active segment entirely via FFmpeg filtergraph.
    No video bytes pass through Python.
    """
    n_seg = frame_end - frame_start + 1

    # Check if pure hold -> delegate to existing FFmpeg hold renderer
    seg_p = p_curve[frame_start:frame_end + 1]
    seg_blur = blur_strength[frame_start:frame_end + 1]
    seg_whip = whip_strength[frame_start:frame_end + 1]
    seg_z = zooms[frame_start:frame_end + 1]

    is_hold = (seg_p > 0.999) & (seg_blur < 0.001) & (seg_whip < 0.001)
    z_range = float(seg_z[is_hold].max() - seg_z[is_hold].min()) if is_hold.any() else 1.0
    is_pure_hold = is_hold.all() and z_range < 0.01 and n_seg > int(fps)

    if is_pure_hold:
        hold_z = float(seg_z[0])
        print(f"     FFmpeg hold: {n_seg} frames at z={hold_z:.2f}", flush=True)
        _render_hold_ffmpeg(
            input_path, output_path, frame_start, frame_end,
            face_data_stable, hold_z, face_side, dest_x_full,
            fps, w, h, enc,
        )
        return

    # Build filtergraph
    fg, sendcmd_path = _build_segment_filtergraph(
        events, face_data_stable, frame_start, frame_end,
        fps, w, h, face_side, dest_x_full, tmp_dir, seg_idx,
    )

    t_start = frame_start / fps
    t_end = (frame_end + 1) / fps

    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(t_start), "-to", str(t_end),
        "-i", input_path,
        "-vf", fg,
    ]

    cmd += [
        "-c:v", enc,
        "-pix_fmt", "yuv420p",
        "-an",
    ]

    # Encoder presets
    if "libx264" in enc:
        cmd += ["-preset", "fast", "-crf", "18"]
    elif "libsvtav1" in enc:
        cmd += ["-preset", "6", "-crf", "28"]
    elif "videotoolbox" in enc:
        cmd += ["-q:v", "65"]
    elif "nvenc" in enc:
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "22"]

    cmd.append(output_path)

    _run_ffmpeg_with_progress(cmd, n_seg, fps)


# ─── Step 8: Nuclear Segment Pipeline ────────────────────────────────────────

def _events_in_range(events, t_start, t_end):
    """Filter events that overlap a time range [t_start, t_end]."""
    result = []
    for ev in events:
        ev_start = ev["start"]
        ev_end = ev["end"]
        if ev_end >= t_start and ev_start <= t_end:
            result.append(ev)
    return result


def _run_segment_pipeline_nuclear(
    input_path, output_path, render_ranges, n_frames, fps,
    face_data_stable, p_curve, zooms,
    blur_strength, whip_strength,
    events, face_side, dest_x_full,
    w, h, enc,
):
    """
    Orchestrate nuclear segment-based rendering:
    stream-copy passthrough, FFmpeg-native active rendering, concat.
    Same segment splitting logic as _run_segment_pipeline but calls
    _render_active_segment_nuclear for active segments.
    """
    import bisect
    tmp_dir = tempfile.mkdtemp(prefix="zb_nuclear_")
    segments = []  # (path, type, frame_start, frame_end)
    seg_idx = 0
    min_hold_frames = int(fps)

    kf_times = _probe_keyframe_times(input_path)
    kf_frames = sorted(set(int(round(t * fps)) for t in kf_times)) if kf_times else []

    def _snap_forward(frame):
        i = bisect.bisect_left(kf_frames, frame)
        return kf_frames[i] if i < len(kf_frames) else None

    def _snap_backward(frame):
        i = bisect.bisect_right(kf_frames, frame) - 1
        return kf_frames[i] if i >= 0 else None

    prev_end = 0
    for rng_start, rng_end in render_ranges:
        # Passthrough before this range
        if rng_start > prev_end:
            pass_start = prev_end
            pass_end = rng_start - 1
            if kf_frames:
                snapped_start = _snap_forward(pass_start)
                snapped_end_kf = _snap_backward(pass_end)
                if (snapped_start is not None and snapped_end_kf is not None
                        and snapped_start < snapped_end_kf):
                    if snapped_start > pass_start:
                        seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                        segments.append((seg_path, "active", pass_start, snapped_start - 1))
                        seg_idx += 1
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_pass.mp4")
                    segments.append((seg_path, "passthrough", snapped_start, snapped_end_kf - 1))
                    seg_idx += 1
                    if snapped_end_kf <= pass_end:
                        seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                        segments.append((seg_path, "active", snapped_end_kf, pass_end))
                        seg_idx += 1
                else:
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                    segments.append((seg_path, "active", pass_start, pass_end))
                    seg_idx += 1
            else:
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                segments.append((seg_path, "active", pass_start, pass_end))
                seg_idx += 1

        # Split active range: find hold sub-region
        seg_p = p_curve[rng_start:rng_end + 1]
        seg_blur = blur_strength[rng_start:rng_end + 1]
        seg_whip = whip_strength[rng_start:rng_end + 1]
        is_hold = (seg_p > 0.999) & (seg_blur < 0.001) & (seg_whip < 0.001)
        hold_indices = np.where(is_hold)[0]
        if len(hold_indices) > min_hold_frames:
            hold_local_start = int(hold_indices[0])
            hold_local_end = int(hold_indices[-1])
            hold_abs_start = rng_start + hold_local_start
            hold_abs_end = rng_start + hold_local_end
            hold_z = zooms[hold_abs_start:hold_abs_end + 1]
            z_const = float(hold_z.max() - hold_z.min()) < 0.01

            if z_const and (hold_local_end - hold_local_start + 1) > min_hold_frames:
                if hold_abs_start > rng_start:
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                    segments.append((seg_path, "active", rng_start, hold_abs_start - 1))
                    seg_idx += 1
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                segments.append((seg_path, "active", hold_abs_start, hold_abs_end))
                seg_idx += 1
                if hold_abs_end < rng_end:
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                    segments.append((seg_path, "active", hold_abs_end + 1, rng_end))
                    seg_idx += 1
                prev_end = rng_end + 1
                continue

        seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
        segments.append((seg_path, "active", rng_start, rng_end))
        seg_idx += 1
        prev_end = rng_end + 1

    # Trailing passthrough
    if prev_end < n_frames:
        pass_start = prev_end
        pass_end = n_frames - 1
        if kf_frames:
            snapped_start = _snap_forward(pass_start)
            if snapped_start is not None and snapped_start <= pass_end:
                if snapped_start > pass_start:
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                    segments.append((seg_path, "active", pass_start, snapped_start - 1))
                    seg_idx += 1
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_pass.mp4")
                segments.append((seg_path, "passthrough", snapped_start, pass_end))
                seg_idx += 1
            else:
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                segments.append((seg_path, "active", pass_start, pass_end))
                seg_idx += 1
        else:
            seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_pass.mp4")
            segments.append((seg_path, "passthrough", pass_start, pass_end))
            seg_idx += 1

    n_pass = sum(1 for _, t, *_ in segments if t == "passthrough")
    n_active = sum(1 for _, t, *_ in segments if t == "active")
    pass_frames = sum(fe - fs + 1 for _, t, fs, fe in segments if t == "passthrough")
    print(f"   Segment pipeline (nuclear): {len(segments)} segments ({n_active} active, {n_pass} passthrough [{pass_frames} frames stream-copy])")

    t0 = time.monotonic()

    # Extract passthrough segments in parallel
    pass_segs = [(s, fs, fe) for s, typ, fs, fe in segments if typ == "passthrough"]
    if pass_segs:
        total_pass_frames = sum(fe - fs + 1 for _, fs, fe in pass_segs)
        def _extract(args):
            path, fs, fe = args
            _extract_passthrough(input_path, path, fs / fps, (fe + 1) / fps, enc)
        with ThreadPoolExecutor(max_workers=min(len(pass_segs), 4)) as pool:
            list(pool.map(_extract, pass_segs))
        print(f"   Passthrough segments: {len(pass_segs)} stream-copied ({total_pass_frames} frames) in {time.monotonic() - t0:.1f}s")

    # Render active segments
    t1 = time.monotonic()
    active_segs = [(s, fs, fe) for s, typ, fs, fe in segments if typ == "active"]
    total_active_frames = sum(fe - fs + 1 for _, fs, fe in active_segs)

    for si, (path, fs, fe) in enumerate(active_segs):
        n_seg = fe - fs + 1
        print(f"   Rendering segment {si+1}/{len(active_segs)}: frames {fs}-{fe} ({n_seg} frames)", flush=True)
        _render_active_segment_nuclear(
            input_path, path, fs, fe,
            events, face_data_stable, p_curve, zooms,
            blur_strength, whip_strength,
            fps, w, h, face_side, dest_x_full,
            enc, tmp_dir, si,
        )

    elapsed_render = time.monotonic() - t1
    print(f"   Active segments: {len(active_segs)} rendered ({total_active_frames} frames) in {elapsed_render:.1f}s ({total_active_frames / max(elapsed_render, 0.01):.1f} fps)")

    # Concat all segments
    segment_paths = [s for s, *_ in segments]
    tmp_concat = os.path.join(tmp_dir, "concat_silent.mp4")
    _concat_segments(segment_paths, tmp_concat)

    # Mux audio
    print("3. Muxing audio ...")
    mux_audio(input_path, tmp_concat, output_path)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total = time.monotonic() - t0
    print(f"   Total segment pipeline (nuclear): {total:.1f}s")
    print(f"Done -> {output_path}")


# ─── Step 9: Main Entry Point ────────────────────────────────────────────────

def create_zoom_bounce_effect_nuclear(
    input_path,
    output_path,
    zoom_max=1.4,
    bounces=None,
    bounce_mode="snap",
    face_side="center",
    overlay_config=None,
    text_config=None,
    fade_mode="band",
    stabilize=0.0,
    stabilize_alpha=0.02,
    debug_labels=False,
    detect_holds=False,
):
    """
    Nuclear (zero-copy FFmpeg) implementation of zoom-bounce effect.
    Same API as create_zoom_bounce_effect().

    Limitations:
    - stabilize != 0 falls back to original
    - Gradient fade approximated (gblur edges vs. per-row color sampling)
    - Overlays deferred to v2
    - debug_labels not supported
    """
    import cv2

    # Fallback for unsupported features
    if stabilize != 0:
        print("   Nuclear: stabilize != 0, falling back to original renderer")
        from zoom_bounce import create_zoom_bounce_effect
        return create_zoom_bounce_effect(
            input_path=input_path,
            output_path=output_path,
            zoom_max=zoom_max,
            bounces=bounces,
            bounce_mode=bounce_mode,
            face_side=face_side,
            overlay_config=overlay_config,
            text_config=text_config,
            fade_mode=fade_mode,
            stabilize=stabilize,
            stabilize_alpha=stabilize_alpha,
            debug_labels=debug_labels,
            detect_holds=detect_holds,
        )

    if bounces is None:
        bounces = [(1.0, 2.5)]
    if overlay_config is None and text_config is not None:
        overlay_config = text_config

    if debug_labels:
        print("   Nuclear: debug_labels not supported in nuclear mode (ignored)")
    if overlay_config:
        print("   Nuclear: overlays deferred to v2 (ignored)")

    print("1. Analyzing face trajectory (nuclear) ...")

    # Quick probe for video metadata
    probe_cap = cv2.VideoCapture(input_path)
    probe_fps = probe_cap.get(cv2.CAP_PROP_FPS)
    probe_n = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    probe_w = int(probe_cap.get(3))
    probe_h = int(probe_cap.get(4))
    probe_cap.release()

    active_ranges = _compute_active_frame_ranges(
        bounces, probe_fps, probe_n, detect_holds=detect_holds,
    )

    if active_ranges is not None:
        detect_frames = sum(e - s + 1 for s, e in active_ranges)
        print(f"   Selective detection: {detect_frames}/{probe_n} frames ({100*detect_frames/max(probe_n,1):.0f}%)")
        raw_data, fps, (w, h) = get_face_data_seek(input_path, active_ranges, probe_n)
    else:
        fps = probe_fps
        w, h = probe_w, probe_h
        default = (w // 2, h // 2, 100, 100)
        raw_data = [default] * probe_n
        print("   No face-dependent events -- skipping detection")

    face_data_stable = smooth_data(raw_data, alpha=stabilize_alpha)
    n_frames = len(face_data_stable)

    # Parse events
    events = _parse_events(bounces, bounce_mode, zoom_max)

    # Build curves (used for hold detection and passthrough logic)
    times, p_curve, zooms = build_bounce_curves(
        n_frames, fps, bounces, bounce_mode, zoom_max,
    )
    blur_strength, blur_n_samples, whip_strength, whip_direction = build_effect_curves(
        n_frames, fps, bounces, bounce_mode, zoom_max,
    )

    # Compute dest_x
    if face_side == "center":
        dest_x_full = w * 0.5
    elif face_side == "left":
        dest_x_full = w * 0.28
    else:
        dest_x_full = w * 0.72

    # Render ranges
    render_ranges = _compute_render_ranges(bounces, fps, n_frames)
    if render_ranges is None:
        print("   No render ranges — nothing to do")
        # Just copy the file
        shutil.copy2(input_path, output_path)
        return

    src_codec = _probe_source_codec(input_path)
    enc = detect_best_encoder(src_codec)

    render_frames = sum(e - s + 1 for s, e in render_ranges)
    print(f"   Render ranges: {len(render_ranges)} range(s), {render_frames}/{n_frames} frames ({100*render_frames/max(n_frames,1):.0f}%)")
    print(f"2. Nuclear segment pipeline ({bounce_mode} mode, {len(bounces)} bounce(s)) ...")

    _run_segment_pipeline_nuclear(
        input_path, output_path, render_ranges, n_frames, fps,
        face_data_stable, p_curve, zooms,
        blur_strength, whip_strength,
        events, face_side, dest_x_full,
        w, h, enc,
    )
