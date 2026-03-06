## When to use

Use `data_animation` when the speaker mentions specific numbers, statistics, percentages, or metrics. The animated counter draws attention to the data point and makes it memorable. Only animate numbers the speaker actually says — never fabricate data.

## Style matching

- **counter** — simple animated number that counts up from `startValue` to `value`. Good for standalone stats.
- **stat-callout** — large animated number with label below and optional delta arrow. Use for key metrics with context (e.g., "Revenue: $2.5M, up 15%").
- **bar** — horizontal bar chart with 1-4 items animating their width. Use when the speaker compares multiple values. Requires `items` prop.

Set `prefix` for currency symbols ($, EUR) and `suffix` for units (%, M, users, etc.). Use `delta` with stat-callout to show positive (green arrow up) or negative (red arrow down) changes.

## When NOT to use

- Never fabricate data — only animate numbers explicitly stated by the speaker.
- Don't use counter style for non-numeric content.
- Don't use bar style with more than 4 items — it gets cramped.
- Avoid during fast-paced segments where the counter animation wouldn't have time to complete.

## Placement

Default to center or upper area of frame (x: 0.1-0.5, y: 0.2-0.5). For bar charts, ensure enough horizontal space (w: 0.3-0.5). Duration should give the counter time to animate (at least 2 seconds) plus time for the viewer to read.
