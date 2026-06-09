# Improve UI Prompt for MotionBloom

Use the Outstanding UI Agent.

## Your Task

Improve the specified UI file into a polished, production-grade interface that reflects MotionBloom's purpose as a medical-tech tool for parents, clinicians, and researchers analyzing hand tremor.

## Before You Start

1. **Read the project structure**:
   - `motionbloom/ui/pyqt_theme.py` — design system, colors, spacing, typography constants
   - `motionbloom/ui/pyqt_components.py` — available UI components and their styling
   - The target file you're improving
   - `motionbloom/ui/pyqt_app.py` — alternative UI patterns for reference

2. **Identify existing patterns**:
   - What colors are in use? (Primary: #E63946, Text: #0F1117, White: #FFFFFF)
   - What components are available? (DuoCard, DuoButton, DuoHeroCard, DuoMetricsCard, DuoVerdictCard)
   - What layout constraints exist? (Camera view + metrics side-by-side on desktop)
   - What functionality must not break? (Hand detection, tremor analysis, live metrics, history)

3. **Understand the goal**:
   - MotionBloom analyzes hand tremor to help identify neurological conditions
   - Users are non-technical (parents, clinicians) and medical professionals
   - The UI must communicate status clearly and display results with confidence
   - The app must feel professional and trustworthy, not generic or clinical-sterile

## Visual Direction

**Modern, calm, premium, medical-tech**

- Clean visual hierarchy with intentional spacing
- Professional typography with good readability
- Purposeful use of color (red for actions, white for clarity)
- Restrained animations and transitions
- Accessibility-first design (WCAG AA contrast minimum)
- Parent and clinician-friendly UX (not overly technical)

## What You're Optimizing For

### High Priority
1. **Layout hierarchy**: Is the score display prominent? Is the Check button obviously the main action?
2. **Spacing consistency**: Are margins and padding uniform and generous?
3. **Typography clarity**: Is text readable, properly weighted, and organized by importance?
4. **Color intentionality**: Is color used purposefully, or scattered randomly?
5. **Component consistency**: Do all UI elements feel like they belong to the same system?
6. **Product clarity**: Is it obvious what MotionBloom does and what the user should do next?

### Medium Priority
7. **Responsive behavior**: Does the layout scale smoothly when the window resizes?
8. **Accessibility**: Do all interactive elements have visible states? Is contrast adequate?
9. **Visual distinctiveness**: Does the UI feel premium and intentional, or generic?

## What NOT to Do

- ❌ Do NOT add fake testimonials, fake metrics, or fake medical claims
- ❌ Do NOT invent new component types; reuse existing components
- ❌ Do NOT add random colors outside the theme system
- ❌ Do NOT use verbose or cluttered placeholder text
- ❌ Do NOT change the backend logic or core functionality
- ❌ Do NOT add decorative noise, random icons, or unnecessary animations
- ❌ Do NOT create generic SaaS layouts
- ❌ Do NOT break the camera view or metrics display
- ❌ Do NOT use low-contrast text combinations

## Constraints

- Keep the existing PyQt6 framework and QSS styling system
- Use only existing components (DuoCard, DuoButton, DuoHeroCard, DuoMetricsCard, DuoVerdictCard)
- All colors must come from `theme.py` constants
- All spacing must use `theme.PADDING_LG`, `theme.PADDING_MD`, `theme.PADDING_SM`
- Typography must use `theme.FONT_SIZE_*` constants
- Do not break existing functionality
- Assume minimum window size of 1500x950px

## How to Validate Your Changes

After you edit the code:

```bash
cd /Users/aharshi/MotionBloomAppVersion/motionbloomtremor
/Users/aharshi/MotionBloomAppVersion/motionbloomtremor/venv/bin/python motionbloom_run.py
```

Then verify:
1. Does the UI look polished and intentional?
2. Are all original features still working? (camera, analysis, metrics, buttons)
3. Is layout clean and hierarchy clear?
4. Is spacing consistent and generous?
5. Is typography readable and well-balanced?
6. Are colors used purposefully?
7. Are button states and status indicators clear?

## Scoring Rubric

After you complete your changes, score the result **honestly and harshly** from 1–10:

| Dimension | Score | Notes |
|-----------|-------|-------|
| Layout hierarchy | _/10 | Is the focal point obvious? Is visual flow clear? |
| Spacing consistency | _/10 | Are margins and padding uniform and intentional? |
| Typography clarity | _/10 | Is text readable? Is the weight hierarchy clear? |
| Color system intentionality | _/10 | Is color used for purpose, not decoration? |
| Component consistency | _/10 | Do all elements feel like one system? |
| Responsive behavior | _/10 | Does the layout adapt smoothly with window resize? |
| Accessibility | _/10 | Is contrast adequate? Are interactive states visible? |
| Visual distinctiveness | _/10 | Does it feel premium and intentional (not generic)? |
| Product clarity | _/10 | Is it obvious what MotionBloom does? What's next? |
| **OVERALL** | **_/10** | Average of above (must be ≥8.5 for production) |

## If Score < 8.5

Generate a specific fix prompt. Be harsh. Example:

```
The UI is functional but looks generic and lacks intentional design.

Specific problems:
1. Hero card (score display) looks flat and uninspiring.
   - Make the score number larger and bolder
   - Add subtle shadow for depth
   - Increase padding to give it breathing room
   
2. Metrics section is a dense data dump.
   - Group metrics into logical sections (Movement, Analysis, Confidence)
   - Add visual separation with cards or subtle dividers
   - Increase line spacing for readability
   
3. Camera frame doesn't stand out.
   - Increase border width from 2px to 3px
   - Ensure it contrasts clearly against background
   
4. Button hierarchy is unclear.
   - Make "Check" button much larger and more prominent
   - Secondary information should visually recede
   
5. Overall feel is clinical, not premium.
   - Add subtle color accents in the metrics display
   - Improve visual rhythm with better spacing
   - Ensure consistent rounded corners and shadows

Edit: motionbloom/ui/pyqt_integrated_app.py

Iterate once more and re-score.
```

## Format Your Response

### Iteration [N]

**Files Changed:**
- [filename]: [what changed and why]

**Improvements Made:**
1. [fix #1]
2. [fix #2]
3. [fix #3]

**Quality Score:**
- Layout hierarchy: X/10
- Spacing: X/10
- Typography: X/10
- Color system: X/10
- Component consistency: X/10
- Responsive: X/10
- Accessibility: X/10
- Distinctiveness: X/10
- Product clarity: X/10
- **Overall: X/10**

**Remaining Issues** (if score < 8.5):
- [issue #1]
- [issue #2]
- [issue #3]

**Test Instructions:**
```bash
cd /Users/aharshi/MotionBloomAppVersion/motionbloomtremor
/Users/aharshi/MotionBloomAppVersion/motionbloomtremor/venv/bin/python motionbloom_run.py
```

**Production Ready?**
[Yes / No] — [reason]

---

## Starting Point

**Target file(s)**: [Specify which file(s) to improve]

**Current state**: [Describe what's wrong or generic]

**Goal**: [What should the UI accomplish and feel like?]

**Visual direction**: [Style inspiration: "Medical-tech premium", "Clean and minimal", "Research-oriented", etc.]

**Constraints**: 
- Use existing components only
- Keep styling in theme system
- Do not break functionality
- Do not add fake content

**Begin your inspection and improvement process now.**
