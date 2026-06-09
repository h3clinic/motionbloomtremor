# MotionBloom UI Quality Rubric

## Purpose
This rubric is used to **objectively score** MotionBloom UI on nine dimensions, identify gaps, and drive iterative improvements toward a target of **8.5+/10 overall**.

The rubric is **honest and harsh**, not generous. Average design gets 5–6. Good design gets 7–8. Exceptional, intentional, production-ready design gets 8.5+.

---

## Scoring Dimensions

### 1. Layout Hierarchy (1–10)

**What it measures**: Is there a clear visual focal point? Do users know where to look first, second, third?

**Excellent (9–10)**
- Score display is visually dominant and immediately attracts attention
- Primary action (Check button) is obviously the next focal point
- Supporting information (metrics, status) is clearly secondary
- Eye naturally flows through the interface in an intended sequence
- No confusion about what's important

**Good (7–8)**
- Score display is prominent
- Check button is clearly the primary action
- Metrics are visually grouped and secondary
- Visual flow is mostly intuitive
- Hierarchy is present but could be stronger

**Adequate (5–6)**
- Score display exists but doesn't command attention
- Check button is present but not obviously primary
- Metrics and other info compete for visual focus
- Visual flow is unclear
- User might not know what to do first

**Poor (3–4)**
- No clear focal point
- All elements compete equally for attention
- Primary action is not visually distinct
- Visual flow is confusing

**Unacceptable (1–2)**
- Hierarchy is random or actively confusing
- Layout looks accidental

---

### 2. Spacing Consistency (1–10)

**What it measures**: Are margins, padding, and gaps uniform and intentional? Is there adequate breathing room?

**Excellent (9–10)**
- All spacing uses theme constants (PADDING_LG, PADDING_MD, PADDING_SM)
- 16px vertical rhythm applied consistently
- Generous margins around key elements
- Visual rhythm is calm and intentional
- No cramped or cluttered sections
- Breathing room is abundant

**Good (7–8)**
- Most spacing is consistent and follows theme
- Adequate breathing room in most places
- Vertical rhythm is mostly 16px-based
- Margins feel intentional
- Some areas could be more generous

**Adequate (5–6)**
- Spacing is present but inconsistent
- Some areas feel cramped
- Breathing room is minimal
- Mix of intentional and arbitrary spacing

**Poor (3–4)**
- Spacing is random or inconsistent
- Many cramped areas
- No breathing room
- Vertical rhythm is absent

**Unacceptable (1–2)**
- Spacing looks accidental or broken

---

### 3. Typography Clarity (1–10)

**What it measures**: Is text readable, well-weighted, and organized by visual importance?

**Excellent (9–10)**
- Clear size hierarchy (XL for titles, Base for body, SM for secondary)
- Font weights are intentional (bold for emphasis, regular for body)
- Line height is adequate for readability (1.5 for body, 1.3 for headers)
- All text is readable at default system font size
- Information hierarchy is obvious from typography alone
- No tiny, hard-to-read text

**Good (7–8)**
- Hierarchy is present (titles, body, secondary)
- Most text is readable
- Font weights are mostly intentional
- Some areas could be larger or clearer
- Hierarchy is mostly obvious

**Adequate (5–6)**
- Text is readable but hierarchy is weak
- Font weights are inconsistent
- Some text is small or hard to read
- Hierarchy is unclear from typography alone

**Poor (3–4)**
- Text is hard to read
- No clear hierarchy
- Font weights are random
- Typography doesn't support information hierarchy

**Unacceptable (1–2)**
- Text is illegible or nearly so

---

### 4. Color System Intentionality (1–10)

**What it measures**: Is color used purposefully and strategically, or scattered randomly?

**Excellent (9–10)**
- ≤5 colors maximum per view
- Primary brand red (#E63946) used sparingly for high-emphasis actions
- All other colors come from the theme system
- Each color use has a clear purpose (action, success, warning, etc.)
- Color supports visual hierarchy and information clarity
- No random or decorative color use
- Contrast is adequate (WCAG AA minimum)

**Good (7–8)**
- 5–7 colors, mostly from theme
- Brand red is used for primary actions
- Most color use is intentional
- Some random or unclear color use
- Contrast is mostly adequate

**Adequate (5–6)**
- More colors than necessary
- Some colors are from theme, some are custom
- Color use is partially intentional
- Some color choices seem arbitrary
- Contrast is adequate but not excellent

**Poor (3–4)**
- Excessive colors (8+)
- Many custom colors outside theme
- Color use is decorative or random
- Contrast may be inadequate

**Unacceptable (1–2)**
- Color is chaotic or unintelligible

---

### 5. Component Consistency (1–10)

**What it measures**: Do all UI elements feel like they belong to the same system? Do they look intentionally designed together?

**Excellent (9–10)**
- All components are from the existing library (DuoCard, DuoButton, etc.)
- Styling is consistent across all instances
- Shadows, borders, rounded corners are uniform
- Interactive states (hover, focus, active) are consistent
- All elements feel like they were designed together

**Good (7–8)**
- Most components are from the library
- Styling is mostly consistent
- Some interactive states may vary
- Elements mostly feel cohesive

**Adequate (5–6)**
- Mix of library and custom components
- Styling is partially consistent
- Some elements feel out of place
- Cohesion is weak

**Poor (3–4)**
- Many custom components
- Inconsistent styling
- Elements feel like they come from different systems

**Unacceptable (1–2)**
- No consistency

---

### 6. Responsive Behavior (1–10)

**What it measures**: Does the layout adapt smoothly when the window is resized? Are proportions maintained?

**Excellent (9–10)**
- Layout scales smoothly from 1500x950 to larger sizes
- Camera aspect ratio is maintained
- Typography scales proportionally
- Spacing scales proportionally
- No broken layouts at any size
- Metrics scroll gracefully if needed

**Good (7–8)**
- Layout scales well in most cases
- Some minor scaling issues
- Proportions are mostly maintained
- No major breakage

**Adequate (5–6)**
- Layout adapts but not smoothly
- Some elements don't scale well
- Proportions may shift unexpectedly

**Poor (3–4)**
- Layout breaks at certain sizes
- Scaling is inconsistent

**Unacceptable (1–2)**
- Layout is broken or unusable at different sizes

---

### 7. Accessibility (1–10)

**What it measures**: Can users with different abilities navigate and use the interface?

**Excellent (9–10)**
- All text meets WCAG AA contrast (4.5:1 minimum)
- All interactive elements have visible focus and hover states
- All buttons and inputs have clear labels
- No information conveyed by color alone
- Semantic HTML/widgets are used correctly
- Text is readable at default system font size
- No flashing or seizure-triggering animations

**Good (7–8)**
- Most text meets WCAG AA
- Most interactive elements have visible states
- Most elements have clear labels
- Contrast is mostly adequate
- Some minor accessibility gaps

**Adequate (5–6)**
- Some text may not meet WCAG AA
- Some interactive states may be unclear
- Some labels missing
- Accessibility is partially addressed

**Poor (3–4)**
- Many accessibility issues
- Contrast problems
- Missing labels or unclear states

**Unacceptable (1–2)**
- Largely inaccessible

---

### 8. Visual Distinctiveness (1–10)

**What it measures**: Does the UI feel premium and intentional, or generic and lazy?

**Excellent (9–10)**
- Design feels polished and intentional
- Careful attention to detail (shadows, corners, transitions)
- Looks like a premium product, not a generic template
- Visual decisions are purposeful
- Feels like it belongs to MotionBloom specifically

**Good (7–8)**
- Design feels intentional
- Most details are polished
- Mostly looks premium
- Some generic elements

**Adequate (5–6)**
- Design feels partially intentional
- Mix of polished and generic elements
- Could be any medical app

**Poor (3–4)**
- Design feels generic
- Looks like it was built from a template
- No distinctive character

**Unacceptable (1–2)**
- Design feels lazy or accidental

---

### 9. Product Clarity (1–10)

**What it measures**: Is it immediately obvious what MotionBloom does? What should the user do next?

**Excellent (9–10)**
- Within 3 seconds, user understands: "This analyzes hand tremor"
- Current status (Ready, Capturing, Analyzing, Results) is always clear
- What to do next is obvious
- Results are presented with context and confidence
- No ambiguity or confusion
- No jargon without explanation

**Good (7–8)**
- Purpose is clear
- Status is mostly clear
- Next steps are mostly obvious
- Some minor confusion possible

**Adequate (5–6)**
- Purpose requires some thought
- Status is sometimes unclear
- User might not know what to do next
- Some confusing language or jargon

**Poor (3–4)**
- Purpose is unclear
- Status is hard to determine
- User doesn't know what to do

**Unacceptable (1–2)**
- App's purpose is opaque

---

## Overall Score Calculation

**Overall = Average of all 9 dimensions**

### Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| 9–10 | Exceptional, production-ready | Ship it |
| 8.5–8.9 | Good, minor polishing needed | Optional final iteration |
| 8–8.4 | Competent, some gaps | One more iteration |
| 7–7.9 | Functional but generic | Multiple iterations needed |
| 6–6.9 | Adequate but rough | Significant redesign required |
| <6 | Poor or broken | Major rework needed |

---

## How to Use This Rubric

### During Development
1. After each code change, launch the app
2. Evaluate all 9 dimensions honestly (not generously)
3. Write down scores and notes for each
4. Identify the weakest 2–3 dimensions
5. Generate a specific fix prompt targeting those weaknesses
6. Repeat until overall score ≥ 8.5

### Example Scoring Session

```
Layout hierarchy: 7/10
- Score display is visible but not dominant
- Check button is primary action but could be larger
- Metrics are secondary but compete visually

Spacing consistency: 6/10
- Uses theme constants but inconsistently
- Some areas feel cramped
- Camera area has good breathing room, metrics are tight

Typography clarity: 7/10
- Size hierarchy is present
- Font weights are mostly intentional
- Some body text could be larger

Color system: 8/10
- Uses theme colors correctly
- Red is reserved for primary action
- Mostly intentional use

Component consistency: 8/10
- All components are from library
- Styling is consistent

Responsive behavior: 7/10
- Scales smoothly mostly
- Some text doesn't scale proportionally

Accessibility: 7/10
- Contrast is good
- Focus states are clear
- Some labels could be more explicit

Visual distinctiveness: 6/10
- Looks functional but generic
- Could feel more premium with better spacing and shadows

Product clarity: 8/10
- Purpose is clear (tremor analysis)
- Status is obvious
- Next steps are clear

OVERALL: 7.1/10

Next iteration focus:
- Increase layout hierarchy (make score display larger and more dominant)
- Add spacing and breathing room to metrics
- Improve visual distinctiveness with better shadows and card styling
```

---

## Common Mistakes to Avoid

❌ **Mistake**: Scoring generously ("It looks okay, so I'll give it an 8")  
✅ **Fix**: Be harsh. 8 means "excellent and nearly production-ready". 7 means "good but has noticeable gaps". 6 means "adequate but generic".

❌ **Mistake**: Using only one or two dimensions  
✅ **Fix**: Evaluate all nine. Weak typography ruins layout hierarchy. Broken spacing kills visual hierarchy.

❌ **Mistake**: Not iterating  
✅ **Fix**: After scoring, write a specific fix prompt and iterate again. UI quality is achieved through repetition, not luck.

❌ **Mistake**: Accepting generic design  
✅ **Fix**: "Adequate" is not good enough. Aim for 8.5+. The difference between 7 and 8.5 is the difference between "looks like any app" and "looks like a premium product".

---

## Reference: Example Scores

### Real App Example #1: Generic Medical Dashboard
- Layout: 6 (no clear focal point)
- Spacing: 5 (cramped metrics table)
- Typography: 5 (hierarchy unclear)
- Color: 6 (looks like a template)
- Components: 5 (mix of styles)
- Responsive: 4 (breaks on resize)
- Accessibility: 6 (contrast okay)
- Distinctiveness: 4 (completely generic)
- Clarity: 6 (purpose is clear, but layout is confusing)
- **Overall: 5.3/10** — Needs major redesign

### Real App Example #2: Polished, Intentional Design
- Layout: 9 (obvious focal point, clear flow)
- Spacing: 9 (consistent, generous, beautiful rhythm)
- Typography: 8 (clear hierarchy, readable)
- Color: 9 (intentional, purposeful, premium)
- Components: 9 (consistent system)
- Responsive: 8 (smooth scaling)
- Accessibility: 9 (excellent contrast, clear states)
- Distinctiveness: 9 (feels premium and intentional)
- Clarity: 9 (immediate understanding, obvious next steps)
- **Overall: 8.8/10** — Production ready

