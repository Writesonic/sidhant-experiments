# Conflict Resolution in video_effects

Two separate mechanisms — one **temporal** (timeline-level, pre-render) and one **spatial** (layout-level, per-frame).

Branch: `temporal-fafo`

---

## 1. Temporal conflict resolution

**File:** `video_effects/activities/validate.py`
**When:** after LLM-generated cues, before render.

### Algorithm (`validate_timeline`)

1. **Clamp** each cue's `start_time`/`end_time` to `[0, duration]`; enforce min 1s width (lines 28–32).
2. **Filter** effects with `confidence < 0.3` (line 36).
3. **Group by `effect_type`** (line 46). Key premise: conflicts only matter *within the same type* — a zoom and a blur at the same moment aren't a conflict because they live in different phases (per `EFFECT_PHASES`).
4. **Overlap detection per group** (lines 52–72): sort by `start_time`, then for each new cue check

   ```
   k.start_time < effect.end_time  AND  effect.start_time < k.end_time
   ```

   (standard interval-overlap test) against already-kept cues. If it overlaps, keep whichever has higher `confidence`; drop the other. Counter `conflicts_resolved` tracks the drops.
5. **Zoom pair validation** (`_validate_zoom_pairs`): walks zoom cues in time order with a `zoomed_in` flag —
   - drops orphan `out`s (no prior `in`),
   - drops duplicate `in`s (already zoomed),
   - propagates `zoom_level` from the paired `in` to its `out` so the animation returns to the same scale.
6. **Sort** by `(phase, start_time)` so the renderer executes phases in order.

### Key snippet

```python
# video_effects/activities/validate.py:56-68
overlapping = [
    k for k in kept
    if k.start_time < effect.end_time and effect.start_time < k.end_time
]
if overlapping:
    for overlap in overlapping:
        if effect.confidence > overlap.confidence:
            kept.remove(overlap)
            kept.append(effect)
            conflicts_resolved += 1
        else:
            conflicts_resolved += 1
```

---

## 2. Spatial conflict avoidance

**File:** `video_effects/remotion/src/lib/spatial.ts`
**When:** at render time inside Remotion components, keeps overlays off the speaker's face.

### Core primitive — `computeOverlap` (lines 22–34)

Intersection-over-own-area between two normalized rects. Returns how much of rect `a` is covered by rect `b`.

```ts
const overlapX = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
const overlapY = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
return aArea > 0 ? (overlapX * overlapY) / aArea : 0;
```

### `useFaceAvoidance` (lines 57–96)

Given the overlay's bounds and the current `FaceFrame`:

1. Compute overlap with the face rect. If `< 0.01`, no push (early exit).
2. Build a **push vector** from face center → overlay center, normalize it.
3. Scale by `strength = overlap * 0.3` — deeper overlap pushes harder.
4. Wrap in a 15-frame `spring()` so the offset eases in smoothly instead of snapping.

### `useFaceAwareLayout` (lines 119–197)

Extends avoidance with anchor modes: `face-right`, `face-left`, `face-above`, `face-below`, `face-beside`.

- `face-beside` picks the side with more room (`spaceRight` vs `spaceLeft`).
- Positions always run through `clampBounds` (hard `[0.02, 0.98]` safe area) before being spring-animated from static bounds toward the face-relative target.
- Then `useZoomCompensation` (lines 98–115) re-maps coordinates through the current zoom affine

  ```
  sx = 0.5 - tx * zoom
  sy = 0.5 - ty * zoom
  adjustedX = zoom * normX + sx
  adjustedY = zoom * normY + sy
  ```

  so overlays track zoomed content instead of sliding off.

---

## Summary

| | Temporal | Spatial |
|---|---|---|
| **Domain** | Discrete cue scheduling | Continuous layout |
| **When** | Pre-render (activity) | Per-frame (Remotion hook) |
| **Unit of conflict** | Same-type cues with time overlap | Overlay rect intersecting face rect |
| **Resolution** | Confidence-ranked drop + phase grouping + interval overlap test | Overlap-proportional repulsion, spring smoothing, clamping, zoom-affine compensation |
