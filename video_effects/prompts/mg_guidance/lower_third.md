## When to use

Use `lower_third` for speaker introductions, topic labels, and segment markers. It works best at the speaker's first appearance or when introducing a new guest/topic. One lower third per speaker introduction is typical; don't repeat for the same speaker unless context changes significantly.

## Style matching

- **slide** — default, feels professional and editorial. The accent bar slides in first, then text follows with a stagger. Good for most content.
- **fade** — simpler, the whole card fades in together. Use for casual or understated content where the slide animation would feel too formal.

Match `accentColor` to the video's color palette. Use the same accent color consistently across all lower thirds in a video.

## When NOT to use

- Don't use for general text overlays or key statements — that's what `animated_title` is for.
- Don't show a lower third while another overlay is already visible in the bottom area.
- Don't use for content that's already clear from context (e.g., a solo creator whose name is in the video title).
- Avoid during zoom effects — the bottom-left placement often falls outside the zoomed viewport.

## Placement

Default to the bottom-left corner (y: 0.75-0.85, x: 0.03-0.10). This template is edge-aligned so it intentionally sits near the screen edge. Always check safe_regions to ensure it doesn't overlap the face.
