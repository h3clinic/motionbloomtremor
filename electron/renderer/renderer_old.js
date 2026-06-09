const METRICS = [
  ["live_score", "Live Score"],
  ["final_score", "Final Score"],
  ["confidence", "Confidence"],
  ["peak_hz", "Peak Hz"],
  ["band_ratio", "Band Ratio"],
  ["amp_mm", "Amplitude"],
  ["snr_db", "SNR"],
  ["tracking_quality", "Tracking"],
];

const metricGrid = document.getElementById("metric-grid");
const logBox = document.getElementById("log-box");
const statusPill = document.getElementById("status-pill");

const values = {};
for (const [key, label] of METRICS) {
  const card = document.createElement("div");
  card.className = "metric";
  card.innerHTML = `<div class="name">${label}</div><div class="value" id="m-${key}">—</div>`;
  metricGrid.appendChild(card);
  values[key] = card.querySelector(".value");
}

function appendLog(message) {
  if (!message) return;
  logBox.textContent = `${new Date().toLocaleTimeString()} ${message}\n${logBox.textContent}`;
}

function setStatus(running, message) {
  statusPill.className = running ? "pill running" : "pill idle";
  statusPill.textContent = running ? "Running" : "Idle";
  if (message) appendLog(message);
}

function applyMetrics(metrics) {
  values.live_score.textContent = metrics.live_score ?? "—";
  values.final_score.textContent = metrics.final_score ?? "—";
  values.confidence.textContent = metrics.confidence ?? "—";
  values.peak_hz.textContent = metrics.peak_hz ?? "—";
  values.band_ratio.textContent = metrics.band_ratio ?? "—";
  values.amp_mm.textContent = metrics.amp_mm ?? "—";
  values.snr_db.textContent = metrics.snr_db ?? "—";
  values.tracking_quality.textContent = metrics.tracking_quality ?? "—";
}

document.getElementById("start-btn").addEventListener("click", async () => {
  const result = await window.motionbloomBridge.start();
  appendLog(result.message);
});

document.getElementById("stop-btn").addEventListener("click", async () => {
  const result = await window.motionbloomBridge.stop();
  appendLog(result.message);
});

window.motionbloomBridge.onEvent((event) => {
  if (!event || !event.type) return;
  if (event.type === "status") {
    setStatus(Boolean(event.running), event.message);
    return;
  }
  if (event.type === "metrics") {
    applyMetrics(event.metrics || {});
    return;
  }
  if (event.type === "log") {
    appendLog(event.message || "");
  }
});

async function setupCameraPreview() {
  const camera = document.getElementById("camera");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    camera.srcObject = stream;
  } catch (error) {
    appendLog(`Camera preview unavailable: ${error.message}`);
    document.getElementById("camera-note").textContent =
      "Camera preview unavailable. Check camera permissions.";
  }
}

setupCameraPreview();
