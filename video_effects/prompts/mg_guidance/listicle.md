## When to use

Use `listicle` when the speaker explicitly lists items, steps, comparisons, or options. The staggered reveal works well when items are mentioned sequentially in speech. Each item should appear roughly when the speaker says it.

## Style matching

- **pop** — items scale in with a bouncy spring. Energetic, good for tips, fun facts, or enthusiastic delivery.
- **slide** — items slide in from the left with a gentle spring. More editorial, good for structured/professional content.

Use `numbered` listStyle for ordered steps or rankings, `bullet` for unordered items, and `none` for minimal/clean look.

## When NOT to use

- Don't fabricate list items the speaker didn't mention — only animate what's actually said.
- Don't use for just 1 item — use `animated_title` instead.
- Don't use for more than 5 items — the screen gets crowded and the stagger takes too long.
- Avoid if the speaker mentions items too quickly (< 0.5s apart) — the stagger won't keep up.

## Placement

Default to the left side of frame (x: 0.1-0.3, y: 0.2-0.6). Items stack vertically, so ensure enough vertical space for all items. Keep the list away from the face region. Duration should be long enough for all items to reveal plus 1-2 seconds of full visibility.
