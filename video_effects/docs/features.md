Code-Generated Motion Graphics

**An AI agent that writes custom animated graphics for your video, from scratch, every time.**

Most video tools give you a library of premade templates and a timeline to drag them onto. Bansi does something different: it listens to what's said in your video, decides what visualizations would help the viewer, and then writes the animation code to create them. Every graphic is one-of-a-kind, built from the actual content of your video.

Two capabilities make this possible:

1. **The Code Runner Agent** -- An AI that analyzes your transcript and writes custom React animation code for charts, diagrams, timelines, code blocks, and more. No premade templates. Real code, generated and validated on the fly.
2. **The Component Template Library** -- A place to save motion graphics you've created (or had the AI create) and reuse them across future videos. Pin them to a workflow, and the AI places them at the right moment automatically.

---

## The Code Runner Agent

### How It Works

The agent operates in three stages: **plan**, **generate**, and **validate**.

**1. Plan -- Six specialist analyzers read your transcript in parallel.**

Each analyzer looks for a different type of content:

| Analyzer | What it looks for | Example output |
|----------|-------------------|----------------|
| Infographics | Statistics, metrics, key numbers | Animated bar chart of revenue growth |
| Diagrams | Processes, flows, relationships | Flowchart of a CI/CD pipeline |
| Timelines | Sequences, histories, progressions | Horizontal timeline of product milestones |
| Quotes | Key statements, takeaways | Styled pull-quote with attribution |
| Code Blocks | Syntax, commands, technical snippets | Highlighted SQL query with line annotations |
| Comparisons | Side-by-side analysis, pros/cons | Two-column versus layout |

Each analyzer proposes 0-3 components with a confidence score. Results are merged, ranked by score, and the top 6 are selected. This means a 10-minute tech tutorial might get a diagram, two code blocks, a comparison table, and a stat callout -- all proposed independently, all ranked against each other.

**2. Generate -- The AI writes custom React/TypeScript animation code for each component.**

This is the part that makes Bansi different from template-based tools. For each planned component, a code generation agent writes a complete animated React component from scratch. The generated code:

- Uses Remotion's animation primitives (spring physics, interpolation, easing curves)
- Reads the video's style preset for fonts, colors, and animation preferences
- Positions itself relative to the speaker's face (never covers them)
- Fades out gracefully in its last half-second
- Uses SVG for data visualization and div layouts for text

The agent has access to a curated set of animation utilities: spring configs (gentle, bouncy, snappy, elastic), math helpers for polar coordinates and arc drawing, color manipulation, and a spatial layout system that understands where the speaker's face is at any given frame.

**3. Validate -- Every generated component is type-checked and test-rendered before it touches your video.**

Validation is a two-step gate:
- **TypeScript type-check** -- The generated code must pass strict TypeScript compilation. No `any` types, no missing imports.
- **Test render** -- Remotion renders a single frame of the component. If it crashes, throws, or produces a blank frame, it fails.

If validation fails, the agent retries up to 3 times, passing the error messages back to the AI so it can fix its own mistakes. If all retries are exhausted, the component falls back to a built-in template (e.g., a failed pie chart becomes an animated bar chart using the built-in data visualization component).

### What It Can Generate

The agent can generate 12 types of visualizations:

- **Charts** -- Pie charts, bar charts, line charts with animated data entry, axis labels, and counting number animations
- **Flowcharts and diagrams** -- Node-and-edge layouts with animated connectors, directional flow
- **Timelines** -- Horizontal event sequences with staggered reveal animations
- **Comparisons** -- Side-by-side versus layouts, pros/cons columns
- **Process flows** -- Numbered step sequences with progressive highlighting
- **Stat dashboards** -- Multiple metrics with counting animations and delta indicators
- **Quotes** -- Styled pull-quotes with attribution, accent borders
- **Code blocks** -- Syntax-highlighted code with optional line highlighting and titles
- **Custom** -- Anything else the AI thinks would work. The type system is open-ended.

Each visualization carries structured data extracted from the transcript. A bar chart gets actual numbers the speaker mentioned. A comparison table gets the actual items being compared. A code block gets the actual syntax being discussed.

### The Creative Agent (Programmer Mode)

Alongside the six specialist analyzers, there's a second path: the creative agent. This is a free-form mode with no predefined categories.

It works in three phases:
1. **Brainstorm** -- The AI reads the transcript and proposes creative visual ideas with no constraints. It can invent anything: a progress ring, a mood meter, a concept map, an animated checklist.
2. **Self-critique** -- It evaluates its own proposals for visual quality, relevance to the content, and technical feasibility. Weak ideas are filtered out.
3. **Generate and validate** -- Surviving ideas go through the same code generation and validation pipeline as the specialist planners.

The creative agent scales with video length: a 1-minute clip gets up to 3 components, a 4-minute video gets up to 15.

**Use case.** A fitness coach walks through a 4-exercise routine. The creative agent proposes an animated exercise card that highlights each movement as the coach explains it, with a progress ring that fills as the routine progresses. None of the six specialist categories would have proposed this -- it's a novel visual that fits the specific content.

---

## Built-In Templates

Not everything needs to be generated from scratch. Four battle-tested templates are always available as building blocks. The code runner uses them as fallbacks when generation fails, and the template library system builds on them.

| Template | What it does | Duration |
|----------|-------------|----------|
| **Animated Title** | Text overlay with fade, slide-in, typewriter, or bounce animation. Configurable font size and color. | 2-5 seconds |
| **Lower Third** | Name and title card with accent bar. Slide or fade entrance. Designed for speaker introductions. | 3-6 seconds |
| **Listicle** | Bullet-point list with staggered pop or slide animations. Up to 5 items. | 3-8 seconds |
| **Data Animation** | Three modes: counting number, stat callout with delta, or animated bar chart. | 2-6 seconds |

These templates are also what generated components fall back to. If a custom pie chart fails validation after 3 retries, it becomes an animated bar chart using the Data Animation template with the same data.

---

## The Component Template Library

The library lets you create, save, and reuse motion graphic components across all your videos.

### Creating Components

Two paths:
- **Conversational** -- Describe what you want in natural language ("a subscribe reminder with my channel colors and a bell icon animation"). The AI generates the React/TypeScript code.
- **Direct** -- Write or edit the component code yourself. The same Remotion API and utility libraries used by the code runner are available.

Both paths produce a live preview in the browser via the built-in Remotion player. You see the animation running in real-time before saving.

### Saving and Reusing

Each saved template includes:
- The component code (TSX)
- Display name, description, and tags for browsing
- Configurable props (text, colors, sizes) that can be filled differently per video
- Spatial hints (where the component prefers to be placed)
- Duration range (how long it should display)

### Pinning to Workflows

The key workflow integration: pin a saved template to a video processing run. When you do:

1. The AI sees the template's description and available props
2. It reads the transcript to find the best moment to place it
3. It fills in episode-specific prop values (e.g., the episode title, the topic name)
4. It positions the component to avoid the speaker's face and other overlays
5. The template renders alongside any generated components

**Use case.** A YouTube channel has a branded "key takeaway" card saved in their library. They pin it to every workflow. For a video about database performance, the AI places it at 4:23 when the speaker summarizes "the three rules of indexing," fills in the takeaway text, and positions it to the right of the speaker's face.

---

## Face-Aware Positioning

Every motion graphic -- generated or from the library -- is positioned with awareness of the speaker's face.

The system tracks faces throughout the video and builds a spatial map: where the face is in each frame, how much space is available on each side, and which regions are "safe" for overlays. Six anchor modes control how components relate to the face:

| Anchor | Behavior |
|--------|----------|
| Static | Fixed position, ignores face entirely |
| Face-right | Positioned to the right of the face |
| Face-left | Positioned to the left of the face |
| Face-above | Above the face |
| Face-below | Below the face |
| Face-beside | Automatically picks whichever side has more space |

When zoom effects are active, overlays compensate for the changed viewport. A graphic that sits at the right edge of the frame at 1x zoom adjusts inward when a 1.5x zoom activates, so it stays visible and clear of the face.

Spatial validation also prevents overlays from colliding with each other. Components are placed in z-index priority order, and each placed component becomes an obstacle for the next one. If no valid position exists, the component is relocated to the nearest free rectangle in the frame.

---

## Human Review

Before any motion graphics are rendered into the final video, you review them in the browser.

The review screen shows:
- Every planned component with its timing, position, and props
- A live Remotion preview that plays the graphics over your video
- Per-component controls to edit props, reposition, or remove
- A text field to reject the entire plan with specific feedback

Rejection sends your feedback back to the AI, which re-plans. Up to five revision rounds. The AI sees exactly what you wrote ("remove the bar chart, the numbers are wrong" or "move the diagram earlier, it should appear when I start explaining the architecture").

**Use case.** The AI generates a comparison table, a flowchart, and a stat callout. The producer previews them and sees the comparison table appears too late -- the speaker has already moved on. They reject with "move the comparison to 1:15-1:25 when I'm actually discussing the tradeoffs." The AI re-plans with corrected timing.

---

## Style System

Every generated component inherits its visual identity from the active style preset. The style controls:

- **Typography** -- Font family, weights for headings/body/emphasis
- **Color palette** -- Text color, secondary color, accent color (3 values that flow into every generated component)
- **Animation preferences** -- Which animation types to prefer (bounce, fade, slide, pop, typewriter) and which to avoid
- **Overlay density** -- How many components to generate (2-3 for sparse styles, 6-10 for high-energy)

Seven presets are available:

| Style | Accent | Vibe | Overlay Density |
|-------|--------|------|-----------------|
| Default | Gold | Balanced | 3-6 |
| Clean Minimal | Copper | Elegant, restrained | 3-4 |
| Bold Energy | Neon green | High-impact, punchy | 6-10 |
| Tech Sleek | Red | Modern, snappy | 4-6 |
| Casual Vlog | Orange | Friendly, relaxed | 3-5 |
| Podcast Pro | Blue | Clean, minimal | 2-3 |
| TikTok Native | Pink | Bold, dense | 6-10 |

The AI auto-detects the best style from the transcript's tone and energy, or you pick one yourself. The selected style is injected into every code generation prompt, so generated components use the right fonts, colors, and animation style without any manual configuration.

---

## Use Cases

**Tech tutorial.** A developer records a 10-minute video explaining microservices. The code runner generates: an architecture diagram showing service boundaries, a comparison table of monolith vs. microservices, and a code block showing the Docker Compose configuration. All positioned to avoid the speaker, styled in Tech Sleek with red accents.

**Podcast clip.** A podcaster extracts a 3-minute highlight. Podcast Pro style gives them a lower-third for the guest's name, a quote card for the best soundbite, and sparse overlays that don't compete with the conversation.

**TikTok explainer.** A creator shoots a 60-second take about compound interest. TikTok Native style generates a punchy stat callout ("$10K becomes $26K in 20 years"), an animated bar chart showing growth over decades, and bold word-highlighted subtitles. High density, high energy.

**Branded series.** A media company produces weekly market updates. They save a branded intro animation, stat callout template, and outro card to the library. Each week, they pin all three to the workflow. The AI fills in that week's numbers and places everything automatically. The producer reviews in the browser, approves in two clicks.

**Education platform.** An instructor records lectures about algorithms. The code runner generates animated flowcharts for each algorithm, step-by-step process visualizations, and Big-O comparison tables -- all extracted from what the instructor actually says on camera. The creative agent adds a custom "complexity meter" that animates alongside the discussion.
