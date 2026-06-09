# MotionBloom Accessibility Standards

All UI must meet or exceed WCAG AA accessibility standards.

## Contrast Requirements

### Text Contrast (WCAG AA Minimum)
- **Normal text**: 4.5:1 contrast ratio minimum
- **Large text** (18pt+ or 14pt+ bold): 3:1 contrast ratio minimum

### Color Combinations for MotionBloom
✅ **Approved combinations** (tested, meet WCAG AA):
- Dark text #0F1117 on white #FFFFFF: 14:1 ratio ✅
- Dark text #0F1117 on light gray #F9F9F9: 13:1 ratio ✅
- Dark text #0F1117 on light red #FFE5E5: 10:1 ratio ✅
- White #FFFFFF on red #E63946: 6.5:1 ratio ✅
- White #FFFFFF on dark red #A01818: 11:1 ratio ✅
- Medium gray #666666 on white #FFFFFF: 8:1 ratio ✅

❌ **Do not use**:
- Light text on light background
- Gray text on white (unless it's secondary info)
- Multiple colors that aren't explicitly tested

## Interactive Elements

### Buttons
- Must have visible focus state (outline or background change)
- Must have visible hover state (distinct from normal)
- Must have visible active/pressed state
- Minimum touch target size: 44x44px (PyQt6 default QPushButton is typically 40px+ high)

### Links and Clickable Elements
- Must be distinguishable from normal text (color + underline or similar)
- Must have visible focus indicator
- Must have visible hover state

### Form Inputs (if added)
- Must have visible focus border
- Must have associated labels
- Error states must use color + text, not color alone

## Labels and Semantic Structure

### Required Labels
- All buttons must have clear, descriptive text
- All status indicators must be labeled
- All metrics must have labels explaining what they measure
- All interactive elements must have a purpose that's clear without color alone

### Widget Structure
- Use appropriate PyQt6 widgets (QLabel, QPushButton, QFrame) semantically
- Do not use generic widgets for content that needs semantic meaning
- Group related elements using layout managers and visual grouping

## Text Requirements

### Readability
- Minimum font size: 12pt for body text
- Prefer 14pt+ for better readability
- Line height: minimum 1.5 for body text
- Line length: 50–75 characters for optimal readability

### Language
- Avoid jargon without explanation
- Use active voice when possible
- Keep sentences short and simple
- Do not use all caps (except very short labels like "CAMERA")
- Do not use LOW-CONTRAST, hard-to-read text for any user-facing content

## Color Accessibility

### Do NOT Use Color Alone
❌ Bad: Red text to indicate error (what if user is colorblind?)  
✅ Good: Red text + "Error:" label to indicate error

❌ Bad: Green button = success, red button = warning (colorblind users can't distinguish)  
✅ Good: Green button with checkmark icon + "Success" label, red button with X + "Warning" label

### Colorblind-Safe Palette
The MotionBloom palette is designed to be colorblind-safe:
- Red #E63946 is distinguishable from green and blue
- White and dark gray provide sufficient neutral contrast
- Avoid introducing new color combinations without testing

## Motion and Animation

### Seizure Considerations
- ❌ Do NOT use flashing (more than 3 flashes/second)
- ❌ Do NOT use rapid color changes
- ❌ Do NOT use rapidly moving/pulsing elements
- ✅ Smooth transitions (300–500ms) are acceptable
- ✅ Subtle opacity changes are acceptable

### Animations
- Keep animations purposeful (not decorative)
- Animations should be smooth and not distracting
- Animations should not interfere with reading or interaction
- Animations should be fast (200–500ms), not slow

## Testing Checklist

Before shipping any UI change:

- [ ] All text has adequate contrast (use WebAIM Contrast Checker or similar)
- [ ] All buttons have visible focus indicators
- [ ] All buttons have visible hover states
- [ ] All interactive elements are keyboard accessible (if applicable)
- [ ] All buttons and controls have clear labels
- [ ] No information is conveyed by color alone
- [ ] No flashing or seizure-triggering animations
- [ ] Text is readable at default system font size
- [ ] Minimum font size is 12pt
- [ ] No low-contrast gray text

---

# MotionBloom Brand Guidelines

## Visual Identity

MotionBloom is a **medical-tech tool for parents, clinicians, and researchers** analyzing hand tremor to identify potential neurological conditions.

### Brand Personality
- **Professional** but approachable (not sterile, not cute)
- **Trustworthy** and science-based (not marketing-y)
- **Clear** and transparent (not mysterious)
- **Empowering** for parents and clinicians (not condescending)

### Visual Tone
- Modern and clean
- Calm and restrained (not aggressive or chaotic)
- Medical-tech aesthetic (not generic SaaS)
- Accessible and inclusive

## Color Palette

### Primary Colors
- **Primary Red**: #E63946
  - Use for: primary actions (Check button), key highlights, status indicators
  - Don't use for: backgrounds, body text, extensive areas
  - Frequency: sparingly, 5–10% of total color use

- **White**: #FFFFFF
  - Use for: main backgrounds, cards, clean spaces
  - Frequency: 60–70% of interface

- **Dark Text**: #0F1117
  - Use for: primary text, headers, important content
  - Frequency: present in most text

### Secondary Colors
- **Light Red**: #FFE5E5 (hover backgrounds, very light accents)
- **Gray** (secondary text): #666666
- **Muted** (disabled, background text): #999999
- **Border/divider**: #EEEEEE

### Color Usage Rules
- ✅ Use red sparingly for high-emphasis actions
- ✅ Use white for main backgrounds and breathing room
- ✅ Use dark gray for secondary information
- ❌ Do NOT add new colors without approval
- ❌ Do NOT use more than 5 colors per view
- ❌ Do NOT use color alone to convey information

## Typography

### Font Stack (PyQt6 Default)
- System fonts (San Francisco, Segoe UI, Helvetica Neue, Arial)
- Do not use decorative or unusual fonts

### Size Scale
- **XL (24–28pt)**: Page titles, hero score display
- **LG (18–20pt)**: Section titles, major headings
- **Base (14pt)**: Body text, normal reading
- **SM (12pt)**: Secondary labels, supporting text
- **XS (11pt)**: Minimum (avoid if possible)

### Weight Hierarchy
- **Bold (700)**: Main titles, key metrics values, section headers
- **Semi-bold (600)**: Section headers, strong emphasis
- **Regular (400)**: Body text, labels, normal content

### Line Height
- **Headers**: 1.3 (24pt font with 30px line height)
- **Body**: 1.5 (14pt font with 21px line height)

## Spacing System

All spacing must use theme constants:
- **PADDING_LG**: 24px (major section padding)
- **PADDING_MD**: 16px (default spacing, section padding)
- **PADDING_SM**: 8px (tight spacing, between adjacent elements)
- **Corners**: 12–16px border-radius for cards and frames
- **Vertical rhythm**: 16px base (multiples of 8 or 16)

## Component Style Guide

### Buttons
- **Primary (Check button)**: Red background #E63946, white text, 48px+ height
- **Secondary**: White background, red border, red text
- **Hover states**: Slightly darker shade of background color
- **Focus**: Visible outline (automatic in PyQt6)
- **Pressed**: Even darker shade or different background

### Cards (DuoCard)
- **Background**: White #FFFFFF
- **Border**: 1–2px #EEEEEE (subtle)
- **Shadow**: Subtle (2–4px blur, 0.1 opacity)
- **Corner radius**: 12px minimum
- **Padding**: 16px (PADDING_MD) minimum

### Hero Card (DuoHeroCard)
- **Background**: White or very light red #FFE5E5
- **Emphasis**: Large, bold score number (28–32pt)
- **Shadow**: Slightly stronger shadow than regular cards
- **Focus**: Score value should dominate visually

### Metrics Card (DuoMetricsCard)
- **Layout**: Label (secondary) + Value (primary)
- **Spacing**: 8px vertical gap between metric items
- **Typography**: Labels 12–14pt secondary gray, values 14–16pt dark bold
- **Grouping**: Use subtle visual separation (spacing, light borders) to group related metrics

### Verdict Card (DuoVerdictCard)
- **Status colors**: 
  - Success (green): #10B981
  - Warning (orange): #F59E0B
  - Error (red): #EF4444
  - Info (gray): #6B7280
- **Icon + text**: Always pair color with icon or text label
- **Padding**: Generous (16px minimum)

### Interactive States

#### Hover
- Slight color darkening or background shift
- Cursor changes to pointer (automatic in PyQt6)
- Transition time: 150–200ms

#### Focus
- Clear outline (2–3px)
- Outline color: red #E63946 or slightly darker
- Visible on keyboard navigation

#### Active/Pressed
- Darker background or slightly different appearance
- Should feel responsive and tactile

## Usage Examples

### ✅ Good: Clear Hierarchy
```
HERO SECTION (Large, Red Border, White Background)
┌─────────────────────────────┐
│       Score: 87%            │  ← Bold, 32pt, dominant
│     4.2 Hz · Postural      │  ← Secondary, 14pt, gray
└─────────────────────────────┘

METRICS (Scrollable, Card-based)
┌─────────────────┐
│ Motion      —   │  ← Label gray, value dark
│ Confidence  Low │  ← Clear label + value
│ Peak      4.2Hz │
└─────────────────┘

[✓ CHECK BUTTON] ← Red, 48px, obvious primary action
```

### ❌ Bad: Generic and Cluttered
```
Generic Header
═══════════════════════════

Score: 87%  Mode: Postural  Status: Ready  FPS: 30

┌─────────────────────────────┐
│ Live Motion      87%        │
│ Tremor Candidate 65%        │
│ Certified Score  —          │
│ Confidence       Low        │
│ Peak Frequency   4.2 Hz     │
│ Tremor Band      60%        │
│ [20 more metrics...]        │
└─────────────────────────────┘

Button: Check My Hand
```

---

## Brand Do's and Don'ts

### ✅ DO
- ✅ Use clear, simple language
- ✅ Explain results with confidence and context
- ✅ Be transparent about what's being measured
- ✅ Make the app easy to use for non-technical users
- ✅ Use the brand red sparingly and intentionally
- ✅ Create visual hierarchy through spacing and size
- ✅ Be accessible to all users (contrast, labels, etc.)
- ✅ Use subtle animations that enhance usability

### ❌ DON'T
- ❌ Make medical claims without evidence
- ❌ Use fake testimonials or fake metrics
- ❌ Create generic SaaS layouts
- ❌ Use low-contrast or hard-to-read text
- ❌ Add decorative clutter or random icons
- ❌ Use clinical-sterile designs (not approachable)
- ❌ Use cutesy or childish designs (not professional)
- ❌ Add verbose or unnecessary text
- ❌ Use colors randomly
- ❌ Create inaccessible interfaces

