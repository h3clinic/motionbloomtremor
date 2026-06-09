# MotionBloom Duolingo-Style UI

## 🎉 What's New

The Electron UI has been completely redesigned with a **Duolingo-inspired gamified interface** while maintaining all the original MotionBloom tremor analysis features.

## 🌸 Key Features

### Visual Design
- **Bright Theme**: Red and white MotionBloom brand colors (goodbye dark mode!)
- **MotionBloom Logo**: Displayed prominently in the header
- **Bloom Mascot**: Animated mascot with emotions (idle, wave, analyzing, cheer, sad)
- **3D Card Effects**: Rounded corners, shadows, hover animations
- **Smooth Animations**: Floating mascot, progress bars with gradients

### Gamification Elements

#### 🔥 Streak Counter
- Tracks consecutive days of tremor checks
- Persists in localStorage
- Orange gradient pill design
- Auto-increments on daily usage

#### 💎 XP System
- Earn XP based on exercise completion:
  - 50 XP: No tremor detected
  - 40 XP: Mild tremor
  - 30 XP: Moderate tremor
  - 20 XP: Strong tremor
- Blue gradient pill
- Saves progress locally

#### ❤️ Lives/Hearts
- 5 hearts maximum
- Visual health indicator
- Red gradient pill
- Decreases if exercise fails

### Exercise System

#### 📋 Four Exercise Modes

1. **Free Check** (👋)
   - Basic tremor analysis without specific posture
   - No timer, continuous analysis
   - Great for quick checks

2. **Touch Nose** (👃)
   - Intention tremor test
   - 3 sec prepare + 6 sec hold
   - Bring index fingertip to nose

3. **Scratch Head** (🤚)
   - Kinetic/postural tremor test
   - 3 sec prepare + 6 sec hold
   - Hand on top of head

4. **Hold Object** (☕)
   - Postural tremor test
   - 3 sec prepare + 8 sec hold
   - Hold cup/phone at chest height with extended arm

#### Exercise Flow
- **Prepare Stage**: Get into position, countdown timer
- **Hold Stage**: Maintain pose while collecting data
- **Complete**: Results with verdict and XP reward
- **Progress Bar**: Visual feedback with green gradient
- **On-Screen Instructions**: Overlay text guides user

### UI Layout

#### Top Header
- MotionBloom logo + branding
- Streak/XP/Hearts pills (always visible)

#### Left Sidebar
- Exercise selection cards (click to switch)
- Active exercise highlighted with red border
- Camera and Reports buttons (for future features)

#### Center Panel
- **Hero Card**: 
  - Bloom mascot (animated float)
  - Dynamic title/subtitle
  - Score circle (color-coded by severity)
- **Camera Feed**: 
  - Live webcam preview
  - Exercise instruction overlay
  - Countdown timer overlay
  - Control buttons (Start/Stop/Complete)
- **Progress Bar**: Shows prepare/hold stage progress

#### Right Sidebar
- **Live Metrics** (8 cards):
  - Motion Score
  - Final Score
  - Confidence
  - Peak Hz
  - Band Ratio
  - Amplitude (mm)
  - SNR (dB)
  - Tracking Quality
- **Session Stats**:
  - Peak Hz (max from session)
  - Avg Score
  - Sample count
- **Event Log**: Timestamped activity feed

## 🔧 Technical Details

### Files Created
- `electron/renderer/index.html` - Main UI structure
- `electron/renderer/styles.css` - Duolingo-inspired CSS
- `electron/renderer/renderer.js` - Exercise logic, gamification, IPC

### Original Files Preserved
- `index_old.html`, `styles_old.css`, `renderer_old.js` (backup of dark theme)

### Backend Integration
- Uses same **TremorAnalysisEngine** as PyQt6 app
- Same **TremorTracker** with MediaPipe hand tracking
- Identical analysis pipeline (FFT, Welch PSD, quality gates)
- Python bridge streams JSON events via stdout

### State Management
- Exercise state machine: idle → prepare → hold → done
- Session data tracking (scores, peak Hz, amplitudes)
- Gamification state in localStorage (persistent)
- Bridge connection status monitoring

## 🎮 How to Use

1. **Select Exercise**: Click any exercise card in left sidebar
2. **Start Analysis**: Click "Start Analysis" button
3. **Follow Instructions**: Read on-screen overlay text
4. **Complete Exercise**: 
   - Free mode: Analyze as long as you want
   - Guided mode: Hold pose during prepare + hold stages
5. **View Results**: Check score circle, session stats, and XP earned
6. **Repeat**: Click "Complete Exercise" to reset for another round

## 🚀 Launch Command

```bash
cd /Users/aharshi/MotionBloomAppVersion/motionbloomtremor/electron
npm start
```

## 📊 Score Interpretation

### Score Circle Colors
- **Green** (<15): No significant tremor 🎉
- **Blue** (15-39): Mild tremor
- **Orange** (40-69): Moderate tremor
- **Red** (70+): Strong tremor

### Metrics Explained
- **Motion/Live Score**: Real-time tremor severity (0-100)
- **Final Score**: Smoothed analysis result
- **Peak Hz**: Dominant tremor frequency (typically 4-12 Hz)
- **Amplitude**: Movement magnitude in millimeters
- **SNR**: Signal-to-noise ratio (higher = cleaner signal)
- **Tracking Quality**: Hand detection confidence

## 🎨 Asset Integration

### Mascot States
- `bloom_idle.png` - Neutral/waiting
- `bloom_wave.png` - Greeting
- `bloom_cheer.png` - Success/celebration
- `bloom_sad.png` - Poor result/error
- `bloom_analyzing.png` - (fallback to idle for now)

### Icons
- `streak_flame.png` - 🔥 for streak counter
- `xp_gem.png` - 💎 for XP counter
- `heart.png` - ❤️ for lives counter
- `motionbloom_logo.png` - Main app logo

All assets in `/motionbloom/assets/duolingo/`

## 📈 Future Enhancements

- [ ] Bidirectional IPC (renderer → Python commands)
- [ ] Exercise pose verification (MediaPipe pose landmarks)
- [ ] Reports view (historical session data)
- [ ] Camera settings (source selection, resolution)
- [ ] Export session results as PDF/JSON
- [ ] Leaderboard (compare with anonymized users)
- [ ] Sound effects (Duolingo-style pings and chimes)
- [ ] Accessibility features (screen reader, high contrast)

## 🔗 Original Features Preserved

✅ All 8 tremor analysis metrics  
✅ TremorAnalysisEngine with adaptive baseline  
✅ MediaPipe hand tracking (landmark 8: index fingertip)  
✅ Optical flow motion gating  
✅ FFT + Welch PSD frequency analysis  
✅ Quality assessment (SNR, tracking confidence)  
✅ Real-time camera preview  
✅ Session data collection  

---

**Built with:** Electron 31.7.7, Python 3.x, MediaPipe 0.10.13, OpenCV, NumPy, SciPy  
**Theme:** MotionBloom red (#e63946) + Duolingo green (#58cc02)  
**Design inspiration:** Duolingo's gamified learning interface
