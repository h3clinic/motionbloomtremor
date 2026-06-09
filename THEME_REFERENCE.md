# MotionBloom Bright Red & White Theme - Quick Reference

## Color Palette

```
PRIMARY (Bright Red)        #E63946  ████ (Vibrant, energetic)
RED_MEDIUM (Hover)          #FF8C8C  ████ (Friendly, interactive)
RED_DARK (Pressed)          #A01818  ████ (Deep, professional)
RED_LIGHT (Background)      #FFE5E5  ████ (Soft, subtle)

WHITE (Surfaces)            #FFFFFF  ████ (Clean, bright)
WHITE_SECONDARY             #F9F9F9  ████ (Off-white variant)

TEXT_PRIMARY (Dark)         #0F1117  ████ (Maximum contrast)
TEXT_SECONDARY (Gray)       #57606A  ████ (Supporting text)
TEXT_MUTED (Light Gray)     #8B949E  ████ (Disabled state)

BORDER_COLOR (Light Pink)   #FECDD3  ████ (Subtle separation)
```

## Component Color Usage

### Header
```
Background:     White (#FFFFFF)
Border-Bottom:  Bright Red (#E63946) - 3px
Title Color:    Bright Red (#E63946)
Subtitle Color: Text Secondary (#57606A)
Status Color:   Bright Red (#E63946)
```

### Buttons
```
Primary:
  Normal:       Bright Red (#E63946) text: white
  Hover:        Medium Red (#FF8C8C) text: white
  Pressed:      Dark Red (#A01818) text: white
  
Secondary:
  Normal:       White, border: Bright Red (#E63946)
  Hover:        Light Red (#FFE5E5), border: Bright Red
  Pressed:      Light Pink (#FECDD3)
```

### Cards
```
Background:     White (#FFFFFF)
Border:         Light Pink (#FECDD3) - 2px
Border Hover:   Bright Red (#E63946) - 2px
Mascot BG:      Light Red (#FFE5E5)
Streak BG:      Bright Red (#E63946)
```

### Input Fields
```
Background:     White (#FFFFFF)
Border Normal:  Light Pink (#FECDD3) - 2px
Border Focus:   Bright Red (#E63946) - 2px
Focus BG:       Off-white (#F9F9F9)
Selection:      Bright Red (#E63946)
```

### Tabs
```
Active:
  Background:   White (#FFFFFF)
  Text:         Bright Red (#E63946) - bold
  Border:       Bright Red (#E63946)
  
Inactive:
  Background:   Off-white (#F9F9F9)
  Text:         Text Secondary (#57606A)
  Border:       Light Pink (#FECDD3)
  
Hover:
  Background:   Light Red (#FFE5E5)
  Text:         Bright Red (#E63946)
```

### Progress & Sliders
```
Background:     Light Red (#FFE5E5)
Fill/Handle:    Bright Red (#E63946)
Handle Border:  White (#FFFFFF)
Handle Hover:   Medium Red (#FF8C8C)
```

## Typography Scale

```
2XL = 32px  Bold    ▌▌▌▌▌  Largest headings
XL  = 24px  Bold    ▌▌▌▌   Major titles
LG  = 18px  Bold    ▌▌▌    Card titles, section headers
MD  = 16px  Medium  ▌▌     Subheadings
BASE= 14px  Normal  ▌      Body text
SM  = 12px  Normal  ▌      Labels, helper text
TINY= 10px  Normal  ▌      Small labels
```

## Spacing Scale

```
2XL = 48px  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  Major sections
XL  = 32px  ▓▓▓▓▓▓▓▓▓▓        Large spacing
LG  = 24px  ▓▓▓▓▓▓▓          Card padding
MD  = 16px  ▓▓▓▓             Layout spacing
SM  = 8px   ▓▓               Component padding
XS  = 4px   ▓                Tight spacing
```

## Border Radius Scale

```
XL  = 24px  ▐▌  Large panels, hero cards
LG  = 16px  ▐▌  Cards, frames
MD  = 12px  ▐▌  Buttons, inputs
SM  = 8px   ▐▌  Small inputs, subtle corners
```

## Status Colors

```
✓ Success   = Bright Red (#E63946)
⚠ Warning  = Warm Red (#FF5252)
✕ Error    = Deep Red (#C1121F)
ℹ Info     = Accent Red (#FF6B6B)
```

## Animation Timings

```
FAST = 150ms   ▓ Button presses, quick transitions
BASE = 300ms   ▓▓ Standard animations, color changes
SLOW = 500ms   ▓▓▓ Pulse effects, entrance animations
```

## Hover States Behavior

```
Card:           Border pink → red (300ms smooth)
Button Primary: Red → lighter red (150ms smooth)
Button Sec:     White → light red bg (150ms smooth)
Input:          Pink border → red border (150ms smooth)
Slider Handle:  Red → medium red (150ms smooth)
Tab:            Inactive → light red bg (300ms smooth)
```

## Key Design Principles

1. **Bright & Bold**: Primary red (#E63946) is eye-catching, energetic
2. **Clean & Modern**: White backgrounds, clear separation
3. **Accessible**: Dark text (#0F1117) meets WCAG AA contrast
4. **Consistent**: Same red used across all interactive elements
5. **Responsive**: Smooth animations provide visual feedback
6. **Professional**: Rounded corners, proper spacing
7. **Intuitive**: Color coding matches user expectations

## Icon Colors

```
Success Icon   (✓): Bright Red (#E63946)
Warning Icon  (⚠): Warm Red (#FF5252)
Error Icon    (✕): Deep Red (#C1121F)
Info Icon     (ℹ): Accent Red (#FF6B6B)
Flame Icon    🔥: Bright Red (#E63946)
Heart Icon    ❤: Bright Red (#E63946)
Lock Icon     🔒: Dark Gray (#8B949E)
Check Icon    ✓: Bright Red (#E63946)
```

## Asset Sizes (AI Generated)

```
Mascots:       256×256px  (bloom_idle, bloom_cheer, etc.)
Icons:         128×128px  (flame, gem, heart, lock, etc.)
Logo:          Scales with header (typically 65px height)
```

## State Transitions

```
Disabled Element:
  Text Color:     Text Muted (#8B949E)
  Background:     Light Gray
  Border:         Light Gray
  Cursor:         Not-allowed

Loading State:
  Spinner Color:  Bright Red (#E63946)
  Background:     Light Red (#FFE5E5)

Active State:
  Background:     Bright Red (#E63946)
  Text:           White
  Border:         Bright Red
```

## Quick Implementation Tips

1. Always use the defined color constants from `pyqt_theme.py`
2. Never hardcode colors - import from theme module
3. Use `theme.PRIMARY` instead of `#E63946`
4. Apply hover effects consistently with 150ms animation
5. Maintain 2px border width on cards
6. Use white surfaces as default background
7. Ensure 24px minimum padding on cards
8. Keep text contrast ratios above 4.5:1

---

*Generated: May 31, 2026*
*Theme: Bright Red & White*
*Status: Production Ready*
