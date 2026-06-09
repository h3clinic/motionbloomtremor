/**
 * In-browser hand motion tracker using @mediapipe/hands.
 * Ported from ~/MotionBloo/src/hooks/useMotionTracking.ts
 * (previous Next.js MotionBloom version).
 * Converted from React hook to a plain ES class for the Electron renderer.
 *
 * Requires tremor-analysis.js to be loaded first (centroid, analyzeTremor).
 * Requires @mediapipe/hands CDN script to be loaded (exposes global `Hands`).
 */

class MotionTracker {
  constructor(config) {
    this.config = Object.assign(
      { stabilityThreshold: 65, sampleWindowSize: 30, tremorSensitivity: 0.3 },
      config || {}
    );
    this._hands = null;
    this._animFrame = null;
    this._posHistory = [];
    this._video = null;
    this._callback = null;
    this.isTracking = false;
  }

  /** Register a callback: fn({ handsDetected, tremor, timestamp }) */
  onUpdate(fn) {
    this._callback = fn;
  }

  /** Start tracking on a <video> element. Safe to call multiple times. */
  async start(videoElement) {
    if (this.isTracking) this.stop();
    this._video = videoElement;
    this.isTracking = true;

    // Hands is set globally by the @mediapipe/hands CDN script.
    // If it hasn't loaded yet, retry after 1 s.
    const HandsClass =
      (typeof Hands !== 'undefined' && Hands) ||
      (window.mediapipe && window.mediapipe.Hands) ||
      null;

    if (!HandsClass) {
      console.warn('[MotionTracker] @mediapipe/hands not ready yet — retrying in 1 s');
      setTimeout(() => { if (this.isTracking) this.start(videoElement); }, 1000);
      return;
    }

    const hands = new HandsClass({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
    });
    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5,
    });
    hands.onResults((results) => this._processResults(results));
    this._hands = hands;

    const sendFrame = async () => {
      if (!this.isTracking) return;
      if (this._video && this._video.readyState >= 2 && this._hands) {
        try {
          await this._hands.send({ image: this._video });
        } catch (_e) {
          // Frame send can fail during cleanup — ignore.
        }
      }
      this._animFrame = requestAnimationFrame(sendFrame);
    };
    this._animFrame = requestAnimationFrame(sendFrame);
  }

  /** Stop tracking and release MediaPipe resources. */
  stop() {
    this.isTracking = false;
    if (this._animFrame) {
      cancelAnimationFrame(this._animFrame);
      this._animFrame = null;
    }
    if (this._hands) {
      try { this._hands.close(); } catch (_e) {}
      this._hands = null;
    }
    this._posHistory = [];
    // Emit a "no hand" state so the UI resets cleanly.
    if (this._callback) {
      this._callback({
        handsDetected: 0,
        tremor: { intensity: 0, frequency: 0, stability: 0, isStable: false },
        timestamp: Date.now(),
      });
    }
  }

  _processResults(results) {
    const handsDetected =
      (results.multiHandLandmarks && results.multiHandLandmarks.length) || 0;
    let tremor = { intensity: 0, frequency: 0, stability: 0, isStable: false };

    if (handsDetected > 0 && results.multiHandLandmarks[0]) {
      const lms = results.multiHandLandmarks[0].map((lm) => ({
        x: lm.x,
        y: lm.y,
        z: lm.z,
      }));
      // Track the centroid of all 21 hand landmarks.
      const center = centroid(lms);
      this._posHistory.push(center);
      if (this._posHistory.length > this.config.sampleWindowSize) {
        this._posHistory = this._posHistory.slice(-this.config.sampleWindowSize);
      }
      tremor = analyzeTremor(this._posHistory, this.config.stabilityThreshold);
    } else {
      this._posHistory = [];
    }

    if (this._callback) {
      this._callback({ handsDetected, tremor, timestamp: Date.now() });
    }
  }
}
