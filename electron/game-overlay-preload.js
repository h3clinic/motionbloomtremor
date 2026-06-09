// game-overlay-preload.js
// Injected into every MotionBloom game-launcher BrowserWindow.
// Adds a floating live-tremor HUD and exposes tremor globals for embedded games.
const { ipcRenderer } = require("electron");

// Globals read by embedded games each frame
window._mbTremorScore = 0;
window._mbTremorFreq  = 5;

window.addEventListener("DOMContentLoaded", () => {
  if (!document.body) return;

  // Build all elements via DOM API so page CSP can't block them.
  function mk(tag, css, text) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  const wrap = mk("div",
    "position:fixed;top:14px;right:14px;z-index:2147483647;" +
    "background:rgba(8,10,18,0.88);backdrop-filter:blur(12px);" +
    "-webkit-backdrop-filter:blur(12px);" +
    "border:1.5px solid rgba(255,255,255,0.13);border-radius:14px;" +
    "padding:12px 16px 10px;color:#fff;user-select:none;" +
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;" +
    "min-width:150px;box-shadow:0 8px 32px rgba(0,0,0,0.6);"
  );

  // Header row: brand + close button
  const header = mk("div",
    "display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;");
  const brand = mk("div",
    "font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;" +
    "color:rgba(255,255,255,.4);",
    "\ud83c\udf38 MotionBloom");
  const closeBtn = mk("button",
    "background:none;border:none;color:rgba(255,255,255,.38);font-size:15px;" +
    "cursor:pointer;padding:0 0 0 8px;line-height:1;",
    "\u00d7");
  closeBtn.addEventListener("click", () => { wrap.style.display = "none"; });
  header.append(brand, closeBtn);

  // Score
  const scoreEl = mk("div",
    "font-size:44px;font-weight:900;color:#22c55e;line-height:1;transition:color .3s;",
    "\u2014");
  const scoreLbl = mk("div",
    "font-size:9px;text-transform:uppercase;letter-spacing:.06em;" +
    "color:rgba(255,255,255,.3);margin-top:1px;",
    "Tremor score");

  // Progress bar
  const track = mk("div",
    "height:3px;background:rgba(255,255,255,.08);border-radius:2px;" +
    "margin-top:9px;overflow:hidden;");
  const bar = mk("div",
    "height:100%;width:0%;background:#22c55e;border-radius:2px;" +
    "transition:width .2s,background .3s;");
  track.appendChild(bar);

  // Status + frequency
  const statusEl = mk("div",
    "font-size:11px;font-weight:600;color:rgba(255,255,255,.5);margin-top:5px;",
    "Show hand to camera\u2026");
  const freqEl = mk("div",
    "font-size:10px;color:rgba(255,255,255,.28);margin-top:2px;",
    "\u2014 Hz");

  [header, scoreEl, scoreLbl, track, statusEl, freqEl].forEach(c => wrap.appendChild(c));
  document.body.appendChild(wrap);

  // ---- Live metrics from Python bridge ----
  ipcRenderer.on("tremor:event", (_e, event) => {
    if (!event) return;

    if (event.type === "metrics") {
      const m = event.metrics || {};
      const score = parseFloat(m.live_score);
      // Update globals for embedded games
      if (!isNaN(score)) {
        window._mbTremorScore = score;
        const col = score < 25 ? "#22c55e" : score < 55 ? "#facc15" : "#e63946";
        scoreEl.textContent = score.toFixed(0);
        scoreEl.style.color = col;
        bar.style.width = Math.min(100, score) + "%";
        bar.style.background = col;
      }
      const freq = parseFloat(m.peak_hz || m.dominant_frequency_hz);
      if (!isNaN(freq)) window._mbTremorFreq = Math.max(1, freq);
      freqEl.textContent = isNaN(freq) ? "\u2014 Hz" : freq.toFixed(2) + " Hz";
      statusEl.textContent =
        m._status || (isNaN(score) ? "Show hand to camera\u2026" : "Tracking \u2713");

    } else if (event.type === "status") {
      statusEl.textContent = event.running ? "Bridge active" : "Bridge stopped";
    }
  });
});
