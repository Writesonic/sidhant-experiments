# Infrastructure Poet — Design Theme

A dark, monochromatic design system built around **technical precision + editorial warmth**. Think terminal meets typeset manifesto. High contrast, minimal noise, one warm accent pulling all the weight.

---

## Color Palette

| Role | Hex | Usage |
|---|---|---|
| Background | `#0a0a0a` | Page / root background |
| Surface | `#0f0f0f` | Cards, panels |
| Surface warm | `#0f0e0b` | Highlighted/active panels (slight warm tint) |
| Border default | `#1a1a1a` | Dividers, subtle separators |
| Border card | `#222222` | Card borders (default state) |
| Border warm | `#2a2518` | Warm-tinted borders on accent panels |
| Text primary | `#e8e0d0` | Body text (slightly warm white) |
| Text secondary | `#b0a090` | Descriptions, supporting copy |
| Text muted | `#776655` | Subdued body, helper text |
| Text dim | `#666666` | Meta labels, timestamps |
| Text ghost | `#555555` | Placeholder-level text |
| Text invisible | `#444444` | Barely-there hints |
| **Accent** | `#c8a96e` | Gold — ALL interactive highlights, icons, borders |
| Accent dim | `#c8a96e44` | Translucent accent (badges, borders) |
| Accent fill | `#c8a96e11` | Accent background wash |
| Accent deep | `#141209` | Active card background (tinted toward accent) |
| Negative | `#ff8888` | "Not you" tags, error-adjacent |
| Negative fill | `#ff666611` | Negative tag background wash |
| Negative border | `#ff666644` | Negative tag border |

---

## Typography

### Font Stack

```css
/* Display / Headings */
font-family: 'Syne', sans-serif;
font-weight: 700 | 800;
letter-spacing: -0.02em;
line-height: 1.1;

/* Body / UI / Code */
font-family: 'IBM Plex Mono', 'Courier New', monospace;
font-weight: 300 | 400 | 500 | 600;
```

Import via Google Fonts:
```
https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne:wght@700;800&display=swap
```

### Type Scale

| Role | Size | Weight | Notes |
|---|---|---|---|
| Hero heading | `clamp(28px, 4vw, 42px)` | 800 | Syne, tight tracking |
| Section heading | `18px` | 700 | Syne |
| Card label | `13px` | 600 | IBM Plex Mono, `letter-spacing: 0.05em` |
| Body | `13px` | 400 | IBM Plex Mono, `line-height: 1.75` |
| Small / meta | `12px` | 400 | IBM Plex Mono, `line-height: 1.6` |
| Label caps | `10–11px` | 400–500 | ALL CAPS, `letter-spacing: 0.15–0.2em` |
| Micro | `10px` | 400 | Ghost text, `letter-spacing: 0.2em` |

### Label Pattern

All section labels use this exact pattern:

```css
font-size: 10px;
color: #555;
letter-spacing: 0.2em;
text-transform: uppercase;
margin-bottom: 12px;
```

---

## Spacing System

Spacing is loose and deliberate — no tight cramming.

| Context | Value |
|---|---|
| Page padding | `32–40px` |
| Section gap | `24–28px` |
| Card padding | `20–24px` |
| Inner card padding | `14–16px` |
| Grid gap | `12px` |
| Inline row gap | `8px` |
| Label → content gap | `6–8px` |

---

## Component Patterns

### Card

```css
border: 1px solid #222;
background: #0f0f0f;
padding: 20px;
cursor: pointer;
transition: all 0.2s ease;
```

**Hover state:**
```css
border-color: #c8a96e;
background: #131313;
```

**Active state:**
```css
border-color: #c8a96e;
background: #141209;
/* + 2px top border accent line */
border-top: 2px solid #c8a96e;
```

### Tag / Badge

```css
/* Default */
display: inline-block;
padding: 3px 10px;
border: 1px solid #333;
font-size: 11px;
color: #888;
letter-spacing: 0.05em;

/* Positive variant */
border-color: #c8a96e44;
color: #c8a96e;
background: #c8a96e11;

/* Negative variant */
border-color: #ff666644;
color: #ff8888;
background: #ff666611;
```

### Frequency Badge

```css
background: #c8a96e22;
border: 1px solid #c8a96e44;
color: #c8a96e;
padding: 2px 8px;
font-size: 10px;
white-space: nowrap;
```

### Accent Quote / Callout Box

```css
background: #0f0e0b;
border: 1px solid #2a2518;
border-left: 3px solid #c8a96e;
padding: 14px 16px;
font-size: 12px;
color: #a09070;
font-style: italic;
line-height: 1.6;
```

### Tab Button

```css
/* Default */
padding: 8px 20px;
border: 1px solid #222;
background: transparent;
color: #666;
font-family: IBM Plex Mono;
font-size: 11px;
letter-spacing: 0.1em;
text-transform: uppercase;
transition: all 0.15s;

/* Active */
background: #c8a96e;
color: #0a0a0a;
border-color: #c8a96e;
font-weight: 600;

/* Hover (inactive only) */
color: #e8e0d0;
border-color: #444;
```

### Divider

```css
border: none;
border-top: 1px solid #1a1a1a;
margin: 24px 0;
```

### Data Row (table-like layout)

```css
display: grid;
grid-template-columns: 1fr 100px 1fr; /* adjust per context */
gap: 12px;
padding: 12px 0;
border-bottom: 1px solid #1a1a1a;
font-size: 12px;
```

---

## Layout Principles

- **Full-bleed dark background** — no white, no card-on-white
- **Tabs flush to left** — no centering of navigation
- **Grid for pillars** — `repeat(auto-fit, minmax(240px, 1fr))` with `12px` gap
- **Detail panel expands below grid** — click to reveal, not modal/overlay
- **No rounded corners** — everything is sharp-edged (`border-radius: 0`)
- **No shadows** — depth comes from color contrast, not elevation

---

## Interaction Model

- Cards: hover → border gold, active → warm background + top accent line
- Tabs: click to swap content panel, active tab fills gold
- Expand/collapse: click pillar card to reveal detail panel below the grid
- Transitions: `all 0.2s ease` on cards, `all 0.15s` on tabs

---

## Design Voice

| Principle | Expression |
|---|---|
| Terminal aesthetic | Monospace everywhere except display headers |
| Editorial weight | Syne at 800 for titles — big, tight, confident |
| Warmth in darkness | `#e8e0d0` text (not pure white), gold accent (not blue) |
| Signal over noise | Ghost-level muted text for anything non-essential |
| No decoration | No gradients, no blur, no glows — just structure and contrast |

---

## Quick Reference CSS Variables

```css
:root {
  --bg: #0a0a0a;
  --surface: #0f0f0f;
  --surface-warm: #0f0e0b;
  --border: #1a1a1a;
  --border-card: #222222;
  --border-warm: #2a2518;

  --text: #e8e0d0;
  --text-secondary: #b0a090;
  --text-muted: #776655;
  --text-dim: #666;
  --text-ghost: #555;

  --accent: #c8a96e;
  --accent-dim: #c8a96e44;
  --accent-fill: #c8a96e11;
  --accent-bg: #141209;

  --negative: #ff8888;
  --negative-fill: #ff666611;
  --negative-border: #ff666644;

  --font-display: 'Syne', sans-serif;
  --font-mono: 'IBM Plex Mono', 'Courier New', monospace;

  --radius: 0;
  --transition-card: all 0.2s ease;
  --transition-tab: all 0.15s;
}
```