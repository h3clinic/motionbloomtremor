/**
 * Tremor analysis utilities.
 * Ported directly from ~/MotionBloo/src/lib/tremor-analysis.ts
 * (previous Next.js MotionBloom version).
 */

function landmarkDistance(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

function centroid(landmarks) {
  const n = landmarks.length;
  if (n === 0) return { x: 0, y: 0, z: 0 };
  const sum = landmarks.reduce(
    (acc, lm) => ({ x: acc.x + lm.x, y: acc.y + lm.y, z: acc.z + lm.z }),
    { x: 0, y: 0, z: 0 }
  );
  return { x: sum.x / n, y: sum.y / n, z: sum.z / n };
}

/**
 * Analyzes a window of hand centroid positions to detect tremor.
 * Returns { intensity (0-1), frequency (Hz), stability (0-100), isStable }.
 */
function analyzeTremor(positionHistory, stabilityThreshold) {
  if (positionHistory.length < 3) {
    return { intensity: 0, frequency: 0, stability: 100, isStable: true };
  }

  // Frame-to-frame deltas
  const deltas = [];
  for (let i = 1; i < positionHistory.length; i++) {
    deltas.push(landmarkDistance(positionHistory[i], positionHistory[i - 1]));
  }

  const meanDelta = deltas.reduce((a, b) => a + b, 0) / deltas.length;
  const variance =
    deltas.reduce((acc, d) => acc + (d - meanDelta) ** 2, 0) / deltas.length;
  const stdDev = Math.sqrt(variance);

  // Count direction reversals for frequency estimate
  let directionChanges = 0;
  for (let i = 2; i < positionHistory.length; i++) {
    const dx1 = positionHistory[i - 1].x - positionHistory[i - 2].x;
    const dx2 = positionHistory[i].x - positionHistory[i - 1].x;
    const dy1 = positionHistory[i - 1].y - positionHistory[i - 2].y;
    const dy2 = positionHistory[i].y - positionHistory[i - 1].y;
    if (dx1 * dx2 < 0 || dy1 * dy2 < 0) directionChanges++;
  }

  // Intensity: stdDev of position delta in normalised coords (typical tremor 0.002–0.02)
  const intensity = Math.min(1, stdDev * 50);

  // Rough frequency: direction changes per frame → Hz assuming ~30 fps
  const frequency = (directionChanges / (positionHistory.length - 2)) * 30 * 0.5;

  // Stability: inverse of mean delta magnitude (0-100)
  const stability = Math.max(0, Math.min(100, 100 - meanDelta * 5000));

  return { intensity, frequency, stability, isStable: stability >= stabilityThreshold };
}

function getFingertips(landmarks) {
  return [4, 8, 12, 16, 20]
    .filter((i) => i < landmarks.length)
    .map((i) => landmarks[i]);
}

function handSpread(landmarks) {
  const tips = getFingertips(landmarks);
  if (tips.length === 0) return 0;
  const center = centroid(tips);
  return tips.map((t) => landmarkDistance(t, center)).reduce((a, b) => a + b, 0) / tips.length;
}
