# Privacy Policy — MotionBloom TremorLab

_Last updated: April 21, 2026_

MotionBloom TremorLab ("the app") is a desktop application that analyzes hand tremor using your device's webcam. This policy describes exactly what data the app touches and what it does with it.

## TL;DR
**Nothing leaves your device.** No video, no images, no landmarks, no scores, no account data. The app has no server, no analytics, no telemetry, and no advertising SDKs.

## What the app accesses

| Resource | Why | Where it goes |
| --- | --- | --- |
| **Camera** | To detect your hand and measure tremor in real time | Frames are processed in RAM and discarded immediately. Never written to disk, never transmitted. |
| **Microphone** _(video playback only)_ | Required by the OS audio pipeline when playing the in-app focus video | Not recorded. |
| **Local file system** | _Optional_ — only if you use "Export CSV" to save your own session metrics | Saved to a location you choose. Nothing is auto-saved. |

## What the app does **not** do
- Does not upload, stream, or transmit any video, audio, image, or biometric data.
- Does not create an account or require a login.
- Does not use third-party analytics (no Google Analytics, Firebase, Mixpanel, etc.).
- Does not show ads.
- Does not read files outside folders you explicitly choose.

## Third-party components
The app is built on open-source libraries (MediaPipe, OpenCV, SciPy, Tkinter, FFmpeg via ffpyplayer). These run entirely locally; none of them phones home in the configuration we ship.

## Children
The app is not directed to children under 13 and does not knowingly collect information from them.

## Medical disclaimer
MotionBloom TremorLab is a **technical demo, not a medical device**. It is not intended to diagnose, treat, cure, or prevent any condition. Do not use it to make clinical decisions.

## Contact
Questions: open an issue at https://github.com/h3clinic/motionbloomtremor/issues
