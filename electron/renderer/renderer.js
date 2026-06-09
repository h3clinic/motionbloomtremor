// MotionBloom renderer.
// Real, locally-stored progress only. No hallucinated/seeded values.

// ---------- LOCAL PERSISTENCE ----------
const Store = {
  KEY: 'motionbloom.store.v1',
  _data: null,
  _load() {
    if (this._data) return this._data;
    try { const raw = localStorage.getItem(this.KEY); this._data = raw ? JSON.parse(raw) : {}; }
    catch (_e) { this._data = {}; }
    return this._data;
  },
  get(key, fallback = null) { const d = this._load(); return key in d ? d[key] : fallback; },
  set(key, value) { const d = this._load(); d[key] = value; try { localStorage.setItem(this.KEY, JSON.stringify(d)); } catch (_e) {} return value; },
  push(listKey, record) { const list = this.get(listKey, []); list.push(record); this.set(listKey, list); return list; }
};

// A session = { ts: epochMillis, score: number(0-100) }. Real data only.
function getSessions() {
  const s = Store.get('sessions', []);
  return Array.isArray(s) ? s.filter(x => x && typeof x.ts === 'number' && typeof x.score === 'number') : [];
}
function recordSession(score, exercise, durationSec) {
  if (typeof score !== 'number' || isNaN(score)) return;
  const rec = { ts: Date.now(), score: Math.round(score) };
  if (typeof exercise === 'string' && exercise.trim()) rec.exercise = exercise.trim();
  if (typeof durationSec === 'number' && isFinite(durationSec) && durationSec > 0) rec.dur = Math.round(durationSec);
  Store.push('sessions', rec);
  renderHome();
  renderAnalytics();
}

// ---------- STATE ----------
const appState = { running: false, sessionData: { samples: [], peakHzList: [], ampMmList: [] } };
let lastMetricsPayloadLogTs = 0;

// ---------- METRICS (used by practice tabs) ----------
const METRICS = [
  ['live_score', 'Motion', '%'], ['final_score', 'Score', '/100'],
  ['confidence', 'Confidence', '%'], ['peak_hz', 'Peak Hz', 'Hz'],
  ['band_ratio', 'Band Ratio', ''], ['amp_mm', 'Amplitude', 'mm'],
  ['snr_db', 'SNR', 'dB'], ['tracking_quality', 'Tracking', '%']
];

const elements = {
  homeView: document.getElementById('home-view'),
  centerView: document.getElementById('center-view'),
  videoView: document.getElementById('video-view'),
  focusedView: document.getElementById('focused-view'),
  camera: document.getElementById('camera'),
  homeStreak: document.getElementById('home-streak'),
  chartCanvas: document.getElementById('score-chart'),
  chartEmpty: document.getElementById('chart-empty'),
  chartToggle: document.getElementById('chart-toggle'),
  avgRange: document.getElementById('avg-range'),
  avgValue: document.getElementById('avg-value'),
  avgSub: document.getElementById('avg-sub'),
  bestEx: document.getElementById('best-ex'),
  bestExSub: document.getElementById('best-ex-sub'),
  worstEx: document.getElementById('worst-ex'),
  worstExSub: document.getElementById('worst-ex-sub'),
  analyticsView: document.getElementById('analytics-view'),
  fpView: document.getElementById('fp-view'),
  gamesView: document.getElementById('games-view'),
  scheduleView: document.getElementById('schedule-view'),
  reportsView: document.getElementById('reports-view'),
  anStreak: document.getElementById('an-streak'),
  anTotal: document.getElementById('an-total'),
  anTotalSub: document.getElementById('an-total-sub'),
  anTime: document.getElementById('an-time'),
  anTimeSub: document.getElementById('an-time-sub'),
  anAvgDur: document.getElementById('an-avgdur'),
  anAvgDurSub: document.getElementById('an-avgdur-sub'),
  anAvgRange: document.getElementById('an-avg-range'),
  anAvgValue: document.getElementById('an-avg-value'),
  anAvgSub: document.getElementById('an-avg-sub'),
  anBestEx: document.getElementById('an-best-ex'),
  anBestExSub: document.getElementById('an-best-ex-sub'),
  anWorstEx: document.getElementById('an-worst-ex'),
  anWorstExSub: document.getElementById('an-worst-ex-sub'),
  anChart: document.getElementById('an-score-chart'),
  anChartEmpty: document.getElementById('an-chart-empty'),
  anChartToggle: document.getElementById('an-chart-toggle'),
  anDist: document.getElementById('an-dist'),
  anTable: document.getElementById('an-table')
};

let chartBucket = 'day';

// ---------- AVERAGES (scaled) ----------
function avgLastN(sessions, n) {
  if (!sessions.length) return null;
  const slice = sessions.slice(-n);
  const sum = slice.reduce((a, s) => a + s.score, 0);
  return { avg: sum / slice.length, count: slice.length };
}
function renderAvg() {
  const sessions = getSessions();
  const sel = elements.avgRange ? elements.avgRange.value : '10';
  const n = sel === 'all' ? Number.MAX_SAFE_INTEGER : parseInt(sel, 10);
  const r = avgLastN(sessions, n);
  if (!elements.avgValue) return;
  if (!r) {
    elements.avgValue.textContent = '\u2014';
    if (elements.avgSub) elements.avgSub.textContent = 'no sessions yet';
    return;
  }
  elements.avgValue.textContent = r.avg.toFixed(1);
  if (elements.avgSub) {
    elements.avgSub.textContent = 'based on ' + r.count + (r.count === 1 ? ' session' : ' sessions');
  }
}

// ---------- STREAK (consecutive days with >= 1 session) ----------
function dayKey(ts) { const d = new Date(ts); return d.getFullYear() + '-' + (d.getMonth() + 1) + '-' + d.getDate(); }
function computeStreak(sessions) {
  if (!sessions.length) return 0;
  const days = new Set(sessions.map(s => dayKey(s.ts)));
  let cursor = new Date();
  if (!days.has(dayKey(cursor.getTime()))) cursor.setDate(cursor.getDate() - 1); // grace: allow yesterday
  let streak = 0;
  while (days.has(dayKey(cursor.getTime()))) { streak++; cursor.setDate(cursor.getDate() - 1); }
  return streak;
}

// ---------- CHART (dependency-free canvas) ----------
function bucketKey(ts, bucket) {
  const d = new Date(ts);
  if (bucket === 'year') return String(d.getFullYear());
  if (bucket === 'month') return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0');
  if (bucket === 'week') {
    const onejan = new Date(d.getFullYear(), 0, 1);
    const week = Math.ceil((((d - onejan) / 86400000) + onejan.getDay() + 1) / 7);
    return d.getFullYear() + '-W' + String(week).padStart(2, '0');
  }
  return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
}
function bucketAverages(sessions, bucket) {
  const map = new Map();
  for (const s of sessions) {
    const k = bucketKey(s.ts, bucket);
    if (!map.has(k)) map.set(k, { sum: 0, n: 0 });
    const e = map.get(k); e.sum += s.score; e.n += 1;
  }
  return [...map.entries()].sort((a, b) => a[0] < b[0] ? -1 : 1)
    .map(([k, e]) => ({ label: k, avg: e.sum / e.n }));
}
function drawChartOn(cv, emptyEl, bucket) {
  if (!cv) return;
  const data = bucketAverages(getSessions(), bucket);
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);

  if (!data.length) {
    if (emptyEl) emptyEl.style.display = 'flex';
    cv.style.display = 'none';
    return;
  }
  if (emptyEl) emptyEl.style.display = 'none';
  cv.style.display = 'block';

  const W = cv.width, H = cv.height;
  const padL = 44, padR = 20, padT = 20, padB = 40;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  // axes + gridlines (0..100)
  ctx.strokeStyle = '#e5e5e5'; ctx.fillStyle = '#afafaf';
  ctx.font = '12px Helvetica Neue, Arial, sans-serif'; ctx.lineWidth = 1;
  for (let v = 0; v <= 100; v += 25) {
    const y = padT + plotH - (v / 100) * plotH;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(String(v), 12, y + 4);
  }

  const n = data.length;
  const x = i => n === 1 ? padL + plotW / 2 : padL + (i / (n - 1)) * plotW;
  const y = v => padT + plotH - (Math.max(0, Math.min(100, v)) / 100) * plotH;

  // line
  ctx.strokeStyle = '#e63946'; ctx.lineWidth = 3; ctx.beginPath();
  data.forEach((d, i) => { const px = x(i), py = y(d.avg); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
  ctx.stroke();

  // points + x labels (thinned to avoid crowding)
  ctx.fillStyle = '#e63946';
  const step = Math.ceil(n / 8);
  data.forEach((d, i) => {
    const px = x(i), py = y(d.avg);
    ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
    if (i % step === 0 || i === n - 1) {
      ctx.fillStyle = '#777';
      ctx.fillText(d.label, px - 18, H - 14);
      ctx.fillStyle = '#e63946';
    }
  });
}
function drawChart() { drawChartOn(elements.chartCanvas, elements.chartEmpty, chartBucket); }

// ---------- BEST / WORST EXERCISE ----------
// Lower tremor score = better. Best = lowest avg score; Worst = highest avg score.
// Only sessions tagged with an exercise are considered (real data only).
function exerciseAverages(sessions) {
  const map = new Map();
  for (const s of sessions) {
    if (!s.exercise) continue;
    if (!map.has(s.exercise)) map.set(s.exercise, { sum: 0, n: 0 });
    const e = map.get(s.exercise); e.sum += s.score; e.n += 1;
  }
  return [...map.entries()].map(([name, e]) => ({ name, avg: e.sum / e.n, count: e.n }));
}
function setExCard(valEl, subEl, item) {
  if (!valEl) return;
  if (!item) {
    valEl.textContent = '\u2014';
    if (subEl) subEl.textContent = 'no sessions yet';
    return;
  }
  valEl.textContent = item.name;
  if (subEl) {
    subEl.textContent = 'avg ' + item.avg.toFixed(1) + ' over ' +
      item.count + (item.count === 1 ? ' session' : ' sessions');
  }
}
function renderBestWorst() {
  const stats = exerciseAverages(getSessions());
  if (!stats.length) {
    setExCard(elements.bestEx, elements.bestExSub, null);
    setExCard(elements.worstEx, elements.worstExSub, null);
    return;
  }
  let best = stats[0], worst = stats[0];
  for (const s of stats) {
    if (s.avg < best.avg) best = s;   // lowest tremor = best
    if (s.avg > worst.avg) worst = s; // highest tremor = worst
  }
  setExCard(elements.bestEx, elements.bestExSub, best);
  setExCard(elements.worstEx, elements.worstExSub, worst);
}

// ---------- HOME RENDER ----------
function renderHome() {
  const sessions = getSessions();
  if (elements.homeStreak) elements.homeStreak.textContent = computeStreak(sessions);
  renderAvg();
  renderBestWorst();
  drawChart();
}

// ---------- TABS ----------
let currentTab = null;
function showTab(tab) {
  // Ignore repeat clicks on the already-active tab. Re-running the enter/leave
  // chain for the same tab was thrashing the shared Python bridge (start then
  // an immediate stop from a sibling leave* helper), which killed the live
  // tracker ~80 ms after it spawned — so the overlay never appeared.
  if (tab === currentTab) return;
  currentTab = tab;
  const isHome = tab === 'home';
  const isVideo = tab === 'video';
  const isFocused = tab === 'focused';
  const isAnalytics = tab === 'analytics';
  const isTest = tab === 'test';
  const isFpGate = tab === 'fpgate';
  const isGames    = tab === 'games';
  const isSchedule = tab === 'schedule';
  const isReports  = tab === 'reports';
  if (elements.homeView) elements.homeView.style.display = isHome ? 'flex' : 'none';
  if (elements.videoView) elements.videoView.style.display = isVideo ? 'flex' : 'none';
  if (elements.focusedView) elements.focusedView.style.display = isFocused ? 'flex' : 'none';
  if (elements.analyticsView) elements.analyticsView.style.display = isAnalytics ? 'flex' : 'none';
  if (elements.fpView) elements.fpView.style.display = isFpGate ? 'grid' : 'none';
  if (elements.gamesView) elements.gamesView.style.display = isGames ? 'flex' : 'none';
  if (elements.scheduleView) elements.scheduleView.style.display = isSchedule ? 'flex' : 'none';
  if (elements.reportsView) elements.reportsView.style.display = isReports ? 'flex' : 'none';
  const testView = document.getElementById('test-view');
  if (testView) testView.style.display = isTest ? 'flex' : 'none';
  if (elements.centerView) elements.centerView.style.display = (isHome || isVideo || isFocused || isAnalytics || isTest || isFpGate || isGames || isSchedule || isReports) ? 'none' : 'flex';
  if (isHome) renderHome();
  if (isVideo) enterVideoPractice(); else leaveVideoPractice();
  if (isFocused) enterFocused(); else leaveFocused();
  if (isTest) enterTest(); else leaveTest();
  if (isFpGate) enterFpGate(); else leaveFpGate();
  if (isAnalytics) renderAnalytics();
  if (isGames) renderGames();
  if (isSchedule && typeof window.renderSchedule === 'function') window.renderSchedule();
  if (isReports  && typeof window.renderReports  === 'function') window.renderReports();
  // Single authoritative owner of the shared bridge lifecycle. The leave*
  // helpers no longer stop it themselves (that caused a stop-after-start kill
  // when sibling tabs share the bridge); we reconcile exactly once here.
  const wantsBridge = isTest || isFocused || isFpGate || (isVideo && gate.loaded);
  if (wantsBridge) startInBrowserTracking();
  else stopInBrowserTracking();
}
document.querySelectorAll('.nav-tab[data-tab]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    showTab(btn.dataset.tab);
  });
});
if (elements.avgRange) {
  elements.avgRange.addEventListener('change', renderAvg);
}
if (elements.chartToggle) {
  elements.chartToggle.querySelectorAll('button[data-bucket]').forEach(b => {
    b.addEventListener('click', () => {
      elements.chartToggle.querySelectorAll('button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      chartBucket = b.dataset.bucket;
      drawChart();
    });
  });
}

// ---------- METRICS / BRIDGE (practice tabs) ----------
function applyMetrics(metrics) {
  if (appState.running) {
    const s = parseFloat(metrics.final_score); if (!isNaN(s)) appState.sessionData.samples.push(s);
  }
}
if (window.motionbloomBridge && window.motionbloomBridge.onEvent) {
  window.motionbloomBridge.onEvent((event) => {
    if (!event || !event.type) return;
    if (event.type === 'status') { console.log('[trace] bridge status: running=' + event.running + ' "' + (event.message||'') + '"'); appState.running = !!event.running; }
    else if (event.type === 'metrics') {
      const mm = event.metrics || {};
      mm._status = event.status_message || '';
      const now = Date.now();
      if (now - lastMetricsPayloadLogTs >= 1000) {
        lastMetricsPayloadLogTs = now;
        console.log('[metrics.payload]', JSON.stringify({
          displayed_tremor_score: mm.displayed_tremor_score,
          tremor_score: mm.tremor_score,
          tremor_score_raw: mm.tremor_score_raw,
          tremor_evidence: mm.tremor_evidence,
          dominant_frequency_hz: mm.dominant_frequency_hz,
          residual_amplitude_px: mm.residual_amplitude_px,
          coherent_dot_count: mm.coherent_dot_count,
          valid_dot_count: mm.valid_dot_count,
          tracking_quality: mm.tracking_quality,
          validity_status: mm.validity_status,
        }));
      }
      applyMetrics(mm);
      gateOnMetrics(mm);
      focusedOnMetrics(mm);
      testOnMetrics(mm);
      fpOnMetrics(mm);
    }
    else if (event.type === 'frame') { if (!window.__frameLogged) { window.__frameLogged = true; console.log('[trace] first frame received: points=' + event.points + ' active=' + testState.active); } testOnFrame(event); }
  });
}

// ---------- CAMERA (practice tabs only) ----------
let cameraStream = null;

// ============================================================
// TOPOLOGY BRIDGE CONTROL (landmark-free Python pipeline)
// OpenCV Lucas-Kanade optical flow + persistent homology, in
// topotremor/topo_bridge.py. Metrics arrive via the bridge onEvent handler
// wired above. Function names kept as start/stopInBrowserTracking so the
// existing practice-tab call sites don't need to change.
// ============================================================
function startInBrowserTracking(_videoEl) {
  console.log('[trace] startInBrowserTracking: motionbloomBridge=' + (!!window.motionbloomBridge) + ' start=' + (!!(window.motionbloomBridge && window.motionbloomBridge.start)));
  if (window.motionbloomBridge && window.motionbloomBridge.start) window.motionbloomBridge.start();
}

function stopInBrowserTracking() {
  if (window.motionbloomBridge && window.motionbloomBridge.stop) window.motionbloomBridge.stop();
}

async function setupCameraPreview() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (_e) { return; /* camera optional */ }
  if (elements.camera) elements.camera.srcObject = cameraStream;
  const pc = document.getElementById('practice-camera');
  if (pc && !pc.srcObject) pc.srcObject = cameraStream;
  const fc = document.getElementById('focused-camera');
  if (fc && !fc.srcObject) fc.srcObject = cameraStream;
}

// ============================================================
// VIDEO PRACTICE - adaptive tremor gate (ported from video_gate.py)
// ============================================================

// Faithful JS port of AdaptiveBaseline (motionbloom/signal.py). Learns each
// user's typical tremor so the pass threshold personalises over time
// (the "continuously adapt through biomech research" requirement).
class AdaptiveBaseline {
  constructor(alpha = 0.03) {
    this.alpha = alpha;
    this.rms = null;
    this.scoreFloor = null;
    this.scoreCeiling = null;
    this.scoreAvg = null;
    this.samples = 0;
  }
  update(rmsAmp, bandRatio, score) {
    this.samples += 1;
    if (this.scoreAvg === null) this.scoreAvg = score;
    else { const a = 0.02; this.scoreAvg = (1 - a) * this.scoreAvg + a * score; }
    const quiet = bandRatio < 0.22 && score < 25;
    if (quiet) {
      if (this.rms === null) this.rms = rmsAmp;
      else { const blended = (1 - this.alpha) * this.rms + this.alpha * rmsAmp; this.rms = Math.min(this.rms, blended); }
      if (this.scoreFloor === null) this.scoreFloor = score;
      else this.scoreFloor = (1 - this.alpha) * this.scoreFloor + this.alpha * score;
    }
    if (this.scoreCeiling === null) this.scoreCeiling = score;
    else { const decay = 0.005; this.scoreCeiling = Math.max((1 - decay) * this.scoreCeiling, score); }
  }
  personalAvgThreshold(margin = 8.0, dflt = 45.0) {
    if (this.scoreAvg === null || this.samples < 20) return dflt;
    return Math.max(25.0, Math.min(90.0, this.scoreAvg + margin));
  }
}

const gate = {
  baseline: new AdaptiveBaseline(),
  threshold: 50,
  failStreak: 0,
  passStreak: 0,
  smooth: null,
  loaded: false,
  pausedByGate: false,
  scores: [],
  els: null
};

function gateEls() {
  if (gate.els) return gate.els;
  gate.els = {
    video: document.getElementById('practice-video'),
    fileInput: document.getElementById('video-file'),
    overlay: document.getElementById('gate-overlay'),
    overlayMsg: document.getElementById('gate-msg'),
    thrVal: document.getElementById('gate-threshold'),
    liveVal: document.getElementById('gate-live'),
    statusPill: document.getElementById('gate-status'),
    empty: document.getElementById('video-empty')
  };
  return gate.els;
}

function gateParsePercent(str) { const n = parseFloat(str); return isNaN(n) ? 0 : n / 100; }
function gateParseNum(str) { const n = parseFloat(str); return isNaN(n) ? 0 : n; }

function gateRecompute() {
  // Personal base threshold from the biomech baseline, then relaxed/tightened
  // by recent fail/pass streaks (ported from video_gate.py).
  let thr = gate.baseline.personalAvgThreshold(8.0, 50.0);
  thr += gate.failStreak * 6.0;  // relax after pauses so users aren't locked out
  thr -= gate.passStreak * 2.0;  // tighten as the user proves steadier
  gate.threshold = Math.max(25, Math.min(90, thr));
  return gate.threshold;
}

function gateShowOverlay(show, thr) {
  const els = gateEls();
  if (!els.overlay) return;
  els.overlay.style.display = show ? 'flex' : 'none';
  if (show && els.overlayMsg) {
    els.overlayMsg.textContent = 'Tremor too high. Steady your hand below ' + thr.toFixed(0) + ' to keep playing.';
  }
}

function gateOnMetrics(m) {
  const els = gateEls();
  if (!gate.loaded || !els.video || !els.video.src) return;
  // No hand tracked -> pause and prompt (0 score must not count as "steady").
  if (!metricsTracked(m)) {
    if (!gate.pausedByGate && !els.video.paused && !els.video.ended) { els.video.pause(); gate.pausedByGate = true; }
    if (gate.pausedByGate && els.overlay) {
      els.overlay.style.display = 'flex';
      if (els.overlayMsg) els.overlayMsg.textContent = (m && m._status) ? m._status : 'Show your hand to the camera to keep playing.';
    }
    if (els.liveVal) els.liveVal.textContent = '—';
    if (els.statusPill) { els.statusPill.textContent = 'Waiting for hand'; els.statusPill.className = 'gate-pill over'; }
    return;
  }
  const score = gateParseNum(m.live_score);
  const band = gateParsePercent(m.band_ratio);
  const amp = gateParseNum(m.amp_mm);
  gate.baseline.update(amp, band, score);
  // Smooth live score to avoid flicker right at the threshold.
  gate.smooth = gate.smooth === null ? score : 0.3 * score + 0.7 * gate.smooth;
  const thr = gateRecompute();
  if (els.thrVal) els.thrVal.textContent = thr.toFixed(0);
  if (els.liveVal) els.liveVal.textContent = gate.smooth.toFixed(0);

  if (gate.smooth > thr) {
    // Tremor too high -> gate (pause) the video.
    if (!gate.pausedByGate && !els.video.paused && !els.video.ended) {
      els.video.pause();
      gate.pausedByGate = true;
      gate.failStreak = Math.min(gate.failStreak + 1, 6);
      gate.passStreak = 0;
    }
    if (gate.pausedByGate) gateShowOverlay(true, thr);
    if (els.statusPill) { els.statusPill.textContent = 'Paused - steady your hand'; els.statusPill.className = 'gate-pill over'; }
  } else {
    // Steady enough -> release the gate.
    if (gate.pausedByGate) {
      gate.pausedByGate = false;
      gate.passStreak = Math.min(gate.passStreak + 1, 10);
      gate.failStreak = Math.max(gate.failStreak - 1, 0);
      gateShowOverlay(false, thr);
      const p = els.video.play(); if (p && p.catch) p.catch(function () {});
    }
    if (els.statusPill) { els.statusPill.textContent = 'Playing - steady'; els.statusPill.className = 'gate-pill ok'; }
  }
  gate.scores.push(score);
}

function gateLoadFile(file) {
  const els = gateEls();
  if (!file || !els.video) return;
  if (els.video.src) { try { URL.revokeObjectURL(els.video.src); } catch (_e) {} }
  els.video.src = URL.createObjectURL(file);
  els.video.style.display = 'block';
  if (els.empty) els.empty.style.display = 'none';
  gate.loaded = true;
  gate.smooth = null;
  gate.scores = [];
  gate.failStreak = 0;
  gate.passStreak = 0;
  gate.pausedByGate = false;
  gateShowOverlay(false, gate.threshold);
  // Begin live tremor tracking so the gate has data to act on.
  startInBrowserTracking(document.getElementById('practice-camera'));
  const p = els.video.play(); if (p && p.catch) p.catch(function () {});
}

function enterVideoPractice() {
  const els = gateEls();
  if (els.fileInput && !els.fileInput._wired) {
    els.fileInput._wired = true;
    els.fileInput.addEventListener('change', function (e) {
      const f = e.target.files && e.target.files[0];
      if (f) gateLoadFile(f);
    });
  }
  if (els.video && !els.video._wired) {
    els.video._wired = true;
    els.video.addEventListener('ended', function () {
      // Record a real, locally-stored session: average tremor across the watch.
      if (gate.scores.length) {
        const avg = gate.scores.reduce(function (a, b) { return a + b; }, 0) / gate.scores.length;
        const dur = (els.video && isFinite(els.video.duration)) ? els.video.duration : null;
        recordSession(avg, 'Video Practice', dur);
      }
      stopInBrowserTracking();
    });
  }
  const cam = document.getElementById('practice-camera');
  if (cam && cameraStream && !cam.srcObject) cam.srcObject = cameraStream;
  const fsBtn = document.getElementById('video-fs');
  const area = document.getElementById('video-practice-area');
  if (fsBtn && area && !fsBtn._wired) {
    fsBtn._wired = true;
    fsBtn.addEventListener('click', function () {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (area.requestFullscreen) area.requestFullscreen().catch(function () {});
    });
  }
  if (gate.loaded) startInBrowserTracking(document.getElementById('practice-camera'));
}

function leaveVideoPractice() {
  const els = gateEls();
  if (els && els.video && !els.video.paused) els.video.pause();
  // Bridge lifecycle is owned by showTab() — do not stop it here.
}

// ============================================================
// FOCUSED PRACTICE - camera + red sidebar that cycles exercises
// ============================================================
// Real exercises ported from motionbloom/exercises.py.
const FOCUSED_EXERCISES = [
  { name: 'Point-and-Hold', hold: 8,
    desc: 'Extend your arm forward and hold your index finger as still as possible on a fixed target. Captures postural tremor — stability, frequency and drift.' },
  { name: 'Finger-to-Nose', hold: 6,
    desc: 'Touch your index finger to your nose, then extend your arm fully outward, and repeat. Tests action / intention tremor, smoothness and overshoot.' },
  { name: 'Touch-the-Dot', hold: 6,
    desc: 'Reach out and tap targets one after another with your fingertip. Game-like accuracy — measures reaction time, miss distance and corrections.' },
  { name: 'Trace a Line', hold: 8,
    desc: 'Slowly trace a straight line with your fingertip, keeping the motion smooth and even. Fine motor control — deviation, jitter and speed consistency.' },
  { name: 'Open-Close Hand', hold: 6,
    desc: 'Repeatedly open and close your hand at a steady pace. Mobility and repetition — speed, range of motion and tremor during transitions.' },
  { name: 'Reach-and-Return', hold: 8,
    desc: 'Move your hand from a starting position to a target and back again. Controlled movement — path efficiency, hesitation and recovery.' }
];
// Steadiness target: lower live tremor = steadier. Hold below this to fill the ring.
const FOCUSED_STEADY_TARGET = 45;

const focused = { active: false, current: null, held: 0, lastTs: 0, smooth: null, scores: [], count: 0, els: null };

function focusedEls() {
  if (focused.els) return focused.els;
  focused.els = {
    camera: document.getElementById('focused-camera'),
    name: document.getElementById('focused-name'),
    desc: document.getElementById('focused-desc'),
    ring: document.getElementById('focused-ring'),
    ringVal: document.getElementById('focused-ring-val'),
    status: document.getElementById('focused-status'),
    live: document.getElementById('focused-live'),
    count: document.getElementById('focused-count'),
    skip: document.getElementById('focused-skip')
  };
  return focused.els;
}

function focusedPickExercise() {
  const pool = FOCUSED_EXERCISES.filter(e => !focused.current || e.name !== focused.current.name);
  const next = pool[Math.floor(Math.random() * pool.length)] || FOCUSED_EXERCISES[0];
  focused.current = next;
  focused.held = 0;
  focused.smooth = null;
  focused.scores = [];
  focused.lastTs = 0;
  const els = focusedEls();
  if (els.name) els.name.textContent = next.name;
  if (els.desc) els.desc.textContent = next.desc;
  focusedSetRing(0);
  if (els.status) els.status.textContent = 'Get into position, then hold steady';
}

function focusedSetRing(frac) {
  const els = focusedEls();
  const pct = Math.max(0, Math.min(1, frac));
  if (els.ring) els.ring.style.background =
    'conic-gradient(#fff ' + (pct * 360).toFixed(0) + 'deg, rgba(255,255,255,.25) 0deg)';
  if (els.ringVal) els.ringVal.textContent = Math.round(pct * 100) + '%';
}

function focusedComplete() {
  const els = focusedEls();
  if (focused.scores.length) {
    const avg = focused.scores.reduce((a, b) => a + b, 0) / focused.scores.length;
    recordSession(avg, focused.current.name, focused.current.hold);
  }
  focused.count += 1;
  if (els.count) els.count.textContent = focused.count;
  if (els.status) els.status.textContent = 'Nice! Next exercise…';
  setTimeout(focusedPickExercise, 900);
}

// A hand is only "tracked" when MediaPipe reports real tracking quality.
// Without this, live_score = 0 (no hand) would be mistaken for "perfectly steady".
function metricsTracked(m) {
  const q = parseFloat(m && m.tracking_quality);
  return !isNaN(q) && q >= 25;
}

function focusedOnMetrics(m) {
  if (!focused.active || !focused.current) return;
  const els = focusedEls();
  const now = Date.now();
  const tracked = metricsTracked(m);

  if (!tracked) {
    // No hand in view yet: don't score, surface the bridge's guidance.
    if (els.live) els.live.textContent = '—';
    if (els.status) els.status.textContent = (m && m._status) ? m._status : 'Show your hand to the camera';
    focused.smooth = null;
    focused.lastTs = now;
    return;
  }

  const score = parseFloat(m.live_score);
  if (!isNaN(score)) {
    focused.smooth = focused.smooth === null ? score : 0.3 * score + 0.7 * focused.smooth;
    if (els.live) els.live.textContent = focused.smooth.toFixed(0);
  }
  const steady = focused.smooth !== null && focused.smooth <= FOCUSED_STEADY_TARGET;
  if (focused.lastTs) {
    const dt = (now - focused.lastTs) / 1000;
    if (steady) {
      focused.held += dt;
      focused.scores.push(focused.smooth);
      if (els.status) els.status.textContent = 'Holding steady… keep going';
    } else {
      focused.held = Math.max(0, focused.held - dt * 0.5); // gentle decay if you wobble
      if (els.status) els.status.textContent = 'Too much tremor — steady your hand';
    }
  }
  focused.lastTs = now;
  focusedSetRing(focused.held / focused.current.hold);
  if (focused.held >= focused.current.hold) { focused.held = focused.current.hold; focusedComplete(); }
}

function enterFocused() {
  const els = focusedEls();
  if (els.camera && cameraStream && !els.camera.srcObject) els.camera.srcObject = cameraStream;
  if (els.skip && !els.skip._wired) { els.skip._wired = true; els.skip.addEventListener('click', focusedPickExercise); }
  if (els.count) els.count.textContent = focused.count;
  focused.active = true;
  if (!focused.current) focusedPickExercise();
  startInBrowserTracking(document.getElementById('focused-camera'));
}

function leaveFocused() {
  focused.active = false;
  focused.lastTs = 0;
  // Bridge lifecycle is owned by showTab() — do not stop it here.
}

// ============================================================
// ANALYTICS - full local-history dashboard (no camera)
// ============================================================
let analyticsBucket = 'day';

function formatDuration(sec) {
  if (typeof sec !== 'number' || !isFinite(sec) || sec <= 0) return '\u2014';
  sec = Math.round(sec);
  if (sec < 60) return sec + 's';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return h + 'h ' + m + 'm';
  return m + 'm ' + s + 's';
}

// Per-exercise aggregate: count, avg score, avg/total duration (real data only).
function exerciseStats(sessions) {
  const map = new Map();
  for (const sess of sessions) {
    if (!sess.exercise) continue;
    if (!map.has(sess.exercise)) map.set(sess.exercise, { name: sess.exercise, count: 0, sumScore: 0, sumDur: 0, durCount: 0 });
    const e = map.get(sess.exercise);
    e.count += 1; e.sumScore += sess.score;
    if (typeof sess.dur === 'number' && sess.dur > 0) { e.sumDur += sess.dur; e.durCount += 1; }
  }
  return [...map.values()].map(e => ({
    name: e.name, count: e.count,
    avgScore: e.sumScore / e.count,
    totalDur: e.sumDur,
    avgDur: e.durCount ? e.sumDur / e.durCount : 0
  })).sort((a, b) => b.count - a.count);
}

function renderAnalyticsDistribution(stats, totalTagged) {
  const host = elements.anDist;
  if (!host) return;
  if (!stats.length) { host.innerHTML = '<div class="an-empty">No exercises completed yet.</div>'; return; }
  host.innerHTML = stats.map(s => {
    const pct = totalTagged ? (s.count / totalTagged) * 100 : 0;
    return '<div class="an-dist-row">' +
      '<div class="an-dist-top"><span class="an-dist-name">' + s.name + '</span>' +
      '<span class="an-dist-pct">' + pct.toFixed(0) + '%</span></div>' +
      '<div class="an-dist-bar"><div class="an-dist-fill" style="width:' + pct.toFixed(1) + '%"></div></div>' +
      '</div>';
  }).join('');
}

function renderAnalyticsTable(stats) {
  const host = elements.anTable;
  if (!host) return;
  if (!stats.length) { host.innerHTML = '<div class="an-empty">No exercises completed yet.</div>'; return; }
  let html = '<div class="an-table-head">' +
    '<span>Exercise</span><span>Sessions</span><span>Avg score</span><span>Avg time</span><span>Total time</span></div>';
  html += stats.map(s =>
    '<div class="an-table-row">' +
    '<span class="an-c-name">' + s.name + '</span>' +
    '<span>' + s.count + '</span>' +
    '<span class="an-c-score">' + s.avgScore.toFixed(1) + '</span>' +
    '<span>' + formatDuration(s.avgDur) + '</span>' +
    '<span>' + formatDuration(s.totalDur) + '</span>' +
    '</div>').join('');
  host.innerHTML = html;
}

function renderAnalytics() {
  if (!elements.analyticsView) return;
  const sessions = getSessions();

  // streak + totals
  if (elements.anStreak) elements.anStreak.textContent = computeStreak(sessions);
  if (elements.anTotal) elements.anTotal.textContent = sessions.length;
  if (elements.anTotalSub) elements.anTotalSub.textContent =
    sessions.length ? 'across all practice modes' : 'no sessions yet';

  const timed = sessions.filter(s => typeof s.dur === 'number' && s.dur > 0);
  const totalDur = timed.reduce((a, s) => a + s.dur, 0);
  if (elements.anTime) elements.anTime.textContent = timed.length ? formatDuration(totalDur) : '\u2014';
  if (elements.anTimeSub) elements.anTimeSub.textContent =
    timed.length ? 'over ' + timed.length + (timed.length === 1 ? ' timed session' : ' timed sessions') : 'no timed sessions yet';
  if (elements.anAvgDur) elements.anAvgDur.textContent = timed.length ? formatDuration(totalDur / timed.length) : '\u2014';
  if (elements.anAvgDurSub) elements.anAvgDurSub.textContent =
    timed.length ? 'average hold/watch length' : 'no timed sessions yet';

  // average score (dropdown)
  const sel = elements.anAvgRange ? elements.anAvgRange.value : 'all';
  const n = sel === 'all' ? Number.MAX_SAFE_INTEGER : parseInt(sel, 10);
  const ar = avgLastN(sessions, n);
  if (elements.anAvgValue) elements.anAvgValue.textContent = ar ? ar.avg.toFixed(1) : '\u2014';
  if (elements.anAvgSub) elements.anAvgSub.textContent =
    ar ? 'based on ' + ar.count + (ar.count === 1 ? ' session' : ' sessions') : 'no sessions yet';

  // best / worst (lower tremor = better)
  const exAvg = exerciseAverages(sessions);
  if (exAvg.length) {
    let best = exAvg[0], worst = exAvg[0];
    for (const s of exAvg) { if (s.avg < best.avg) best = s; if (s.avg > worst.avg) worst = s; }
    setExCard(elements.anBestEx, elements.anBestExSub, best);
    setExCard(elements.anWorstEx, elements.anWorstExSub, worst);
  } else {
    setExCard(elements.anBestEx, elements.anBestExSub, null);
    setExCard(elements.anWorstEx, elements.anWorstExSub, null);
  }

  // chart
  drawChartOn(elements.anChart, elements.anChartEmpty, analyticsBucket);

  // exercise distribution + per-exercise breakdown
  const stats = exerciseStats(sessions);
  const totalTagged = stats.reduce((a, s) => a + s.count, 0);
  renderAnalyticsDistribution(stats, totalTagged);
  renderAnalyticsTable(stats);
}

// wiring for analytics controls
if (elements.anAvgRange) elements.anAvgRange.addEventListener('change', renderAnalytics);
if (elements.anChartToggle) {
  elements.anChartToggle.querySelectorAll('button[data-bucket]').forEach(b => {
    b.addEventListener('click', () => {
      elements.anChartToggle.querySelectorAll('button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      analyticsBucket = b.dataset.bucket;
      drawChartOn(elements.anChart, elements.anChartEmpty, analyticsBucket);
    });
  });
}

// ============================================================
// HAND TEST TAB - live camera + topology readout (landmark-free)
// Driven entirely by the Python topology bridge metrics (no MediaPipe).
// ============================================================
const testState = { active: false, frameShown: false };

// Show the bridge's annotated frame (tracked optical-flow points drawn on the
// hand) so the user can literally SEE the model following their hand.
function testOnFrame(ev) {
  if (!testState.active || !ev || !ev.jpg) return;
  const img = document.getElementById('test-frame');
  if (!img) return;
  img.src = 'data:image/jpeg;base64,' + ev.jpg;
  if (!testState.frameShown) { img.style.display = 'block'; testState.frameShown = true; }
  const pill = document.getElementById('test-hand-pill');
  if (pill) {
    const ls = ev.landscape || {};
    const conf = (ls.landscape_confidence !== undefined) ? Math.round(ls.landscape_confidence * 100) : 0;
    const tracking = !!ls.weighted_state && (ev.points || 0) > 0;
    pill.textContent = tracking
      ? ('\u270b Tracked ' + conf + '% \u00b7 ' + (ev.points || 0) + ' pts')
      : 'Searching for hand\u2026';
  }
}

// Draw the live time-delay embedding x(t) vs x(t+τ): the actual phase-space
// "axis" the persistent-homology stage operates on. A rhythmic tremor traces a
// LOOP here; noise/steadiness traces a blob.
function drawEmbedding(signal, delay) {
  const cvs = document.getElementById('test-embed');
  if (!cvs) return;
  const ctx = cvs.getContext('2d');
  const W = cvs.width, H = cvs.height, pad = 16;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d0d1a'; ctx.fillRect(0, 0, W, H);

  // Axes through the centre.
  ctx.strokeStyle = 'rgba(255,255,255,.18)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, H / 2); ctx.lineTo(W - pad, H / 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(W / 2, pad); ctx.lineTo(W / 2, H - pad); ctx.stroke();

  const d = Math.max(1, parseInt(delay, 10) || 1);
  if (!Array.isArray(signal) || signal.length < d + 4) return;

  const xs = [], ys = [];
  for (let i = 0; i + d < signal.length; i++) { xs.push(signal[i]); ys.push(signal[i + d]); }
  let mx = 1e-6;
  for (let i = 0; i < xs.length; i++) mx = Math.max(mx, Math.abs(xs[i]), Math.abs(ys[i]));
  const px = (v) => W / 2 + (v / mx) * (W / 2 - pad);
  const py = (v) => H / 2 - (v / mx) * (H / 2 - pad);

  // Trajectory.
  ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 1.5; ctx.beginPath();
  for (let i = 0; i < xs.length; i++) { const X = px(xs[i]), Y = py(ys[i]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
  ctx.stroke();

  // Current position (the moving head of the trace).
  const last = xs.length - 1;
  ctx.fillStyle = '#facc15'; ctx.beginPath(); ctx.arc(px(xs[last]), py(ys[last]), 3, 0, Math.PI * 2); ctx.fill();
}

function fmtNum(x, d) {
  d = (d === undefined) ? 2 : d;
  return (typeof x === 'number' && isFinite(x)) ? x.toFixed(d) : (x == null ? '\u2014' : String(x));
}

function firstFiniteNumber(values) {
  for (const value of values || []) {
    const parsed = (typeof value === 'number') ? value : parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

// Debug decomposition: separates tremor EVIDENCE (what the dot field senses)
// from TRUST (coherence × coverage × tracking) so a suppressed score still
// reveals whether tremor was detected-but-distrusted vs. simply absent.
function renderTestDebug(m, trusted) {
  const dbg = document.getElementById('test-debug');
  if (!dbg) return;
  if (!m || !m.debug_mode) { dbg.style.display = 'none'; return; }
  const ir = m.invalid_reason_counts || {};
  const irStr = Object.keys(ir).length
    ? Object.entries(ir).map(([k, v]) => `${k}:${v}`).join('  ')
    : '(none)';
  const tq = (parseFloat(m.tracking_quality) || 0) / 100;
  const rawScore = firstFiniteNumber([m.tremor_score_raw, m.raw_evidence_score, m.tremor_evidence != null ? (100 * m.tremor_evidence) : null]) || 0;
  const displayedScore = firstFiniteNumber([m.displayed_tremor_score, m.tremor_score]);
  const freq = firstFiniteNumber([m.dominant_frequency_hz, m.dominant_tremor_frequency_hz, m.global_tremor_frequency_hz, m.peak_hz_num]);
  const ampPx = firstFiniteNumber([m.residual_amplitude_px, m.global_tremor_amplitude, m.residual_rms]);
  const coherent = (m.coherent_dot_count !== undefined) ? m.coherent_dot_count : 0;
  const validDots = (m.valid_dot_count !== undefined) ? m.valid_dot_count : 0;
  const trackingQ = firstFiniteNumber([m.tracking_quality_num, parseFloat(m.tracking_quality)]);
  const validity = m.validity_status || '\u2014';
  dbg.style.display = 'block';
  dbg.textContent =
`EVIDENCE vs TRUST\n` +
`  Raw evidence : ${rawScore.toFixed(0)}/100   (dot_score ${fmtNum(m.global_dot_score)})\n` +
`  Trust        : ${m.trust_score || 0}/100   = coh ${fmtNum(m.coherence_factor)} × cov ${fmtNum(m.coverage_factor)} × tq ${fmtNum(tq)}\n` +
`  FINAL        : ${(displayedScore == null ? 0 : displayedScore.toFixed(0))}/100   ${trusted ? 'TRUSTED' : 'UNTRUSTED'} · ${validity}\n\n` +
`MEASUREMENTS\n` +
`  raw ${rawScore.toFixed(0)}  displayed ${(displayedScore == null ? '\u2014' : displayedScore.toFixed(0))}  freq ${(freq == null ? '\u2014' : freq.toFixed(2))}Hz  amp ${(ampPx == null ? '\u2014' : ampPx.toFixed(3))}px\n` +
`  coherent ${coherent}  valid ${validDots}  tracking ${(trackingQ == null ? '\u2014' : trackingQ.toFixed(0))}%  validity ${validity}\n\n` +
`DOTS  total ${m.total_dot_count || 0}  valid ${m.valid_dot_count || 0}/${m.min_valid_dots || 0}\n` +
`  invalid ${m.warming_dot_count || 0}  stable ${m.stable_dot_count || 0}  noise ${m.tracking_noise_dot_count || 0}  relatch ${m.relatching_dot_count || 0}\n` +
`  candidate ${m.candidate_dot_count || 0}  confirmed ${m.tremor_confirmed_count || 0}  coherent ${m.coherent_dot_count || 0}\n\n` +
`AMPLITUDE px  med ${fmtNum(m.median_dot_amplitude)}  p75 ${fmtNum(m.p75_dot_amplitude)}  p90 ${fmtNum(m.p90_dot_amplitude)}  max ${fmtNum(m.max_dot_amplitude)}\n` +
`SPECTRUM(moving)  bpr ${fmtNum(m.median_band_power_ratio)}  prom ${fmtNum(m.median_prominence)}  flat ${fmtNum(m.median_flatness)}\n` +
`INVALID  ${irStr}`;
}

function testOnMetrics(m) {
  if (!testState.active || !m) return;
  const pill = document.getElementById('test-hand-pill');
  const h1Pill = document.getElementById('test-fps-pill');
  const trackBar = document.getElementById('test-bar-stability');   // repurposed: tracking
  const scoreBar = document.getElementById('test-bar-intensity');   // tremor score (red)
  const trackVal = document.getElementById('test-val-stability');
  const scoreVal = document.getElementById('test-val-intensity');
  const freqVal = document.getElementById('test-val-freq');
  const statusEl = document.getElementById('test-status');
  const scoreNum = document.getElementById('test-score-num');
  const scoreSub = document.getElementById('test-score-sub');

  const h1 = (m.h1_lifetime !== undefined) ? parseFloat(m.h1_lifetime) : NaN;
  if (h1Pill) h1Pill.textContent = isNaN(h1) ? 'H1 —' : ('H1 loop ' + h1.toFixed(2));

  const tracked = parseFloat(m.tracking_quality); // "NN%"
  const hasSignal = Array.isArray(m.signal) && m.signal.length > 4;
  const hasMotion = (!isNaN(tracked) && tracked >= 12) || hasSignal;

  if (!hasMotion) {
    if (pill) { pill.textContent = m._status || 'Point camera at the hand'; pill.style.background = 'rgba(0,0,0,.55)'; }
    if (trackBar) trackBar.style.width = '0%';
    if (scoreBar) scoreBar.style.width = '0%';
    if (trackVal) trackVal.textContent = '—';
    if (scoreVal) scoreVal.textContent = '—';
    if (freqVal) freqVal.textContent = '—';
    if (statusEl) statusEl.textContent = m._status || 'Waiting for trackable texture';
    if (scoreNum) { scoreNum.textContent = '—'; scoreNum.style.color = 'rgba(255,255,255,.3)'; }
    if (scoreSub) scoreSub.textContent = 'No motion tracked';
    drawEmbedding(m.signal, m.embed_delay);
    renderTestDebug(m, false);
    return;
  }

  const displayedScore = firstFiniteNumber([m.displayed_tremor_score, m.tremor_score]);
  const rawScore = firstFiniteNumber([m.tremor_score_raw, m.raw_evidence_score, m.tremor_evidence != null ? (100 * m.tremor_evidence) : null]);
  const chosenScore = (displayedScore != null) ? displayedScore : rawScore;
  const s = (chosenScore == null) ? null : Math.max(0, Math.min(100, chosenScore));
  const scoreColor = s < 20 ? '#22c55e' : s < 45 ? '#facc15' : '#e63946';

  // Trust gate: a tremor score is only real when validity === VALID. Under
  // FLOW_LAG / RELATCHING / LANDSCAPE_DIFFUSE / LOW_POINTS the residual is a
  // tracking artifact, so we show the tracking problem and DO NOT show a number.
  const trusted = (m.tremor_trusted !== false);
  const validity = m.validity_status || 'UNKNOWN';
  const isTrackingProblem = (validity === 'FLOW_LAG' || validity === 'RELATCHING' ||
                             validity === 'LANDSCAPE_DIFFUSE' || validity === 'LOW_POINTS');

  if (pill) {
    if (isTrackingProblem) {
      pill.textContent = (validity === 'FLOW_LAG') ? '\u26a0 Tracking lag'
        : (validity === 'RELATCHING') ? '\u26a0 Re-acquiring hand'
        : (validity === 'LANDSCAPE_DIFFUSE') ? '\u26a0 Hold hand still'
        : '\u26a0 Low texture';
      pill.style.background = 'rgba(234,179,8,.78)';
    } else {
      pill.textContent = 'Tracking motion \u2713';
      pill.style.background = 'rgba(34,197,94,.7)';
    }
  }
  if (trackBar) trackBar.style.width = (m.tracking_quality || '0%');
  if (trackVal) trackVal.textContent = m.tracking_quality || '\u2014';
  const freqNum = firstFiniteNumber([m.dominant_frequency_hz, m.dominant_tremor_frequency_hz, m.global_tremor_frequency_hz, m.peak_hz_num]);
  if (freqVal) freqVal.textContent = (freqNum == null) ? '\u2014' : `${freqNum.toFixed(2)} Hz`;

  if (trusted && s != null) {
    if (scoreBar) scoreBar.style.width = s.toFixed(0) + '%';
    if (scoreVal) scoreVal.textContent = s.toFixed(0);
    if (scoreNum) { scoreNum.textContent = s.toFixed(0); scoreNum.style.color = scoreColor; }
    // Per-dot context: how many dots agree, and on which finger / palm.
    const coherentN = (m.coherent_dot_count !== undefined) ? m.coherent_dot_count : 0;
    const confirmedN = (m.tremor_confirmed_count !== undefined) ? m.tremor_confirmed_count : 0;
    const topRegion = pickTopRegion(m.region_tremor);
    let sub;
    if (s < 20) {
      sub = '\u2705 Steady';
    } else if (s > 45) {
      sub = '\u26a0\ufe0f High tremor';
      if (coherentN >= 3) sub += ` \u00b7 ${coherentN} dots @ ${m.global_tremor_frequency_hz || '?'}Hz`;
      if (topRegion) sub += ` \u00b7 ${topRegion}`;
    } else {
      sub = '\ud83e\udd1a Some motion';
      if (confirmedN >= 1 && topRegion) sub += ` \u00b7 ${topRegion}`;
    }
    if (scoreSub) scoreSub.textContent = sub;
  } else {
    // Untrusted: never display a tracking artifact as a trusted tremor number.
    // In DEBUG we still REVEAL the raw evidence (amber, flagged UNTRUSTED) so a
    // detected-but-distrusted tremor is visible instead of reading as zero.
    const rawEv = firstFiniteNumber([m.tremor_score_raw, m.raw_evidence_score, m.tremor_evidence != null ? (100 * m.tremor_evidence) : null]);
    if (rawEv != null && rawEv > 0) {
      if (scoreBar) scoreBar.style.width = rawEv.toFixed(0) + '%';
      if (scoreVal) scoreVal.textContent = rawEv.toFixed(0);
      if (scoreNum) { scoreNum.textContent = rawEv.toFixed(0); scoreNum.style.color = '#f59e0b'; }
      const ampNum = firstFiniteNumber([m.residual_amplitude_px, m.global_tremor_amplitude, m.residual_rms]);
      if (scoreSub) {
        scoreSub.textContent = `UNTRUSTED · ${validity || 'evidence only'}` +
          ((freqNum != null) ? ` · ${freqNum.toFixed(2)}Hz` : '') +
          ((ampNum != null) ? ` · A ${ampNum.toFixed(3)}px` : '');
      }
    } else {
      if (scoreBar) scoreBar.style.width = '0%';
      if (scoreVal) scoreVal.textContent = '\u2014';
      if (scoreNum) { scoreNum.textContent = '\u2014'; scoreNum.style.color = 'rgba(255,255,255,.35)'; }
      if (scoreSub) scoreSub.textContent = (validity === 'FLOW_LAG') ? 'No measurement yet · tracking lag'
        : (validity === 'RELATCHING') ? 'Tremor paused \u00b7 re-acquiring'
        : (validity === 'LANDSCAPE_DIFFUSE') ? 'Tremor paused \u00b7 hold still'
        : 'No measurement yet';
    }
  }

  if (statusEl) {
    // 3D Hand State Landscape + tracking diagnostics. Tracking metrics
    // (flow lag, relatch rate) are the GATE; tremor is shown only when trusted.
    const macroScore = (m.macro_motion_score !== undefined) ? m.macro_motion_score : 0;
    const tremorScore = (m.tremor_score !== undefined) ? m.tremor_score : 0;
    const tConf = (m.tremor_confidence !== undefined) ? Math.round(m.tremor_confidence * 100) : 0;
    const flowLag = (m.flow_lag_px !== undefined) ? m.flow_lag_px : '\u2014';
    const relatch = (m.relatch_rate !== undefined) ? Math.round(m.relatch_rate * 100) : 0;
    const ls = m.landscape || {};
    const nCand = (ls.n_candidates !== undefined) ? ls.n_candidates : 0;
    const uncertainty = (ls.uncertainty_px !== undefined) ? ls.uncertainty_px : '\u2014';

    let statusLine = m._status || (isNaN(h1) ? 'Analyzing\u2026' : ('H1 loop ' + h1.toFixed(2)));
    const trustTag = trusted ? `Tremor:${tremorScore}% (conf ${tConf}%)` : 'Tremor:untrusted';
    // Dense residual-dot field summary.
    const validDots = (m.valid_dot_count !== undefined) ? m.valid_dot_count : 0;
    const totalDots = (m.total_dot_count !== undefined) ? m.total_dot_count : 0;
    const cohDots = (m.coherent_dot_count !== undefined) ? m.coherent_dot_count : 0;
    const confDots = (m.tremor_confirmed_count !== undefined) ? m.tremor_confirmed_count : 0;
    const noiseDots = (m.noise_dot_count !== undefined) ? m.noise_dot_count : 0;
    const minValid = (m.min_valid_dots !== undefined) ? m.min_valid_dots : 30;
    const gFreq = (trusted && m.dominant_tremor_frequency_hz) ? m.dominant_tremor_frequency_hz : 0;
    const topkAmp = (trusted && m.topk_dot_amplitude) ? m.topk_dot_amplitude : 0;
    const sparseTag = (validDots < minValid) ? ' \u26a0sparse' : '';
    const dotInfo = ` | Dots ${validDots}/${totalDots} (v${validDots}\u2265${minValid}${sparseTag} \u00b7 c${cohDots} \u00b7 \u2713${confDots} \u00b7 n${noiseDots})` +
                    (gFreq ? ` @ ${gFreq}Hz` : '') +
                    (topkAmp ? ` \u00b7 A${(typeof topkAmp === 'number' ? topkAmp.toFixed(2) : topkAmp)}px` : '');
    const layerInfo = ` [${validity} | Macro:${macroScore}% | lag:${flowLag}px | relatch:${relatch}% | States:${nCand} | \u03c3:${uncertainty}px | ${trustTag}${dotInfo}]`;

    statusEl.textContent = statusLine + layerInfo;
  }
  drawEmbedding(m.signal, m.embed_delay);
  renderTestDebug(m, trusted);
}

// Return the highest-scoring region label ("index finger", "palm", …) from the
// per-dot region_tremor map, or null when there is no localised tremor.
function pickTopRegion(regionMap) {
  if (!regionMap || typeof regionMap !== 'object') return null;
  let best = null, bestScore = 0;
  for (const [name, info] of Object.entries(regionMap)) {
    const sc = (info && info.score) ? info.score : 0;
    if (sc > bestScore) { bestScore = sc; best = name; }
  }
  if (!best || bestScore <= 0) return null;
  const label = (best === 'palm') ? 'palm' : (best + ' finger');
  return label;
}

function enterTest() {
  console.log('[trace] enterTest called; active=' + testState.active);
  if (testState.active) return;
  testState.active = true;
  testState.frameShown = false;
  const video = document.getElementById('test-camera');
  if (video && cameraStream && !video.srcObject) video.srcObject = cameraStream;
  // The annotated bridge frame will replace the plain camera once it arrives.
  const img = document.getElementById('test-frame');
  if (img) { img.style.display = 'none'; img.removeAttribute('src'); }
  const canvas = document.getElementById('test-canvas');
  if (canvas) { const c = canvas.getContext('2d'); c.clearRect(0, 0, canvas.width, canvas.height); }
  drawEmbedding(null, 1); // draw empty axes immediately
  const statusEl = document.getElementById('test-status');
  if (statusEl) statusEl.textContent = 'Starting topology analysis…';
  startInBrowserTracking(); // start the Python topology bridge
}

function leaveTest() {
  if (!testState.active) return;
  testState.active = false;
  testState.frameShown = false;
  const img = document.getElementById('test-frame');
  if (img) { img.style.display = 'none'; img.removeAttribute('src'); }
  // Bridge lifecycle is owned by showTab() — do not stop it here.
}

// ============================================================
// FIRST-PERSON TREMOR GATE MODE (original clean-room implementation)
// ============================================================
const fp = {
  active: false,
  running: false,
  rafId: 0,
  lastTs: 0,
  spawnTimer: 0,
  tremorSmooth: null,
  threshold: 50,
  failStreak: 0,
  passStreak: 0,
  score: 0,
  passed: 0,
  missed: 0,
  health: 100,
  gates: [],
  baseline: new AdaptiveBaseline(),
  els: null
};

function fpEls() {
  if (fp.els) return fp.els;
  fp.els = {
    canvas: document.getElementById('fp-canvas'),
    status: document.getElementById('fp-status'),
    threshold: document.getElementById('fp-threshold'),
    live: document.getElementById('fp-live'),
    score: document.getElementById('fp-score'),
    passed: document.getElementById('fp-passed'),
    missed: document.getElementById('fp-missed'),
    health: document.getElementById('fp-health'),
    start: document.getElementById('fp-start'),
    reset: document.getElementById('fp-reset')
  };
  return fp.els;
}

function fpRecomputeThreshold() {
  let thr = fp.baseline.personalAvgThreshold(8.0, 50.0);
  thr += fp.failStreak * 6.0;
  thr -= fp.passStreak * 2.0;
  fp.threshold = Math.max(25, Math.min(90, thr));
  return fp.threshold;
}

function fpReset() {
  fp.score = 0;
  fp.passed = 0;
  fp.missed = 0;
  fp.health = 100;
  fp.gates = [];
  fp.spawnTimer = 0;
  fp.tremorSmooth = null;
  fp.failStreak = 0;
  fp.passStreak = 0;
  fp.threshold = 50;
  fp.baseline = new AdaptiveBaseline();
  const els = fpEls();
  if (els.score) els.score.textContent = '0';
  if (els.passed) els.passed.textContent = '0';
  if (els.missed) els.missed.textContent = '0';
  if (els.health) els.health.textContent = '100';
  if (els.threshold) els.threshold.textContent = 'Threshold: 50';
  if (els.live) els.live.textContent = 'Live tremor: —';
  if (els.status) els.status.textContent = 'Ready';
}

function fpSpawnGate() {
  const lane = [-1, 0, 1][Math.floor(Math.random() * 3)];
  fp.gates.push({ z: 1.15, lane, resolved: false });
}

function fpDraw() {
  const els = fpEls();
  const cv = els.canvas;
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width;
  const H = cv.height;
  const horizon = H * 0.34;

  const sky = ctx.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0, '#0b1020');
  sky.addColorStop(0.55, '#141a2a');
  sky.addColorStop(1, '#0b0c10');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = 'rgba(255,255,255,.18)';
  for (let i = 0; i < 7; i++) {
    const t = i / 6;
    const y = horizon + (H - horizon) * t * t;
    ctx.beginPath();
    ctx.moveTo(W * 0.12 + t * W * 0.18, y);
    ctx.lineTo(W * 0.88 - t * W * 0.18, y);
    ctx.stroke();
  }

  fp.gates.forEach((g) => {
    const scale = 1 / Math.max(0.08, g.z);
    const y = horizon + (H - horizon) * (1 - 1 / (1 + scale));
    const laneOffset = g.lane * (W * 0.11) / (g.z + 0.2);
    const gateW = Math.min(W * 0.42, W * 0.14 * scale);
    const gateH = Math.min(H * 0.55, H * 0.18 * scale);
    ctx.strokeStyle = g.resolved ? 'rgba(88,204,2,.55)' : 'rgba(230,57,70,.9)';
    ctx.lineWidth = Math.max(2, 6 / g.z);
    ctx.strokeRect(W / 2 - gateW / 2 + laneOffset, y - gateH, gateW, gateH);
  });

  ctx.strokeStyle = 'rgba(255,255,255,.75)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(W / 2 - 10, H / 2);
  ctx.lineTo(W / 2 + 10, H / 2);
  ctx.moveTo(W / 2, H / 2 - 10);
  ctx.lineTo(W / 2, H / 2 + 10);
  ctx.stroke();
}

function fpTick(ts) {
  if (!fp.active || !fp.running) return;
  if (!fp.lastTs) fp.lastTs = ts;
  const dt = Math.min(0.05, (ts - fp.lastTs) / 1000);
  fp.lastTs = ts;

  fp.spawnTimer += dt;
  if (fp.spawnTimer >= 2.4) {
    fp.spawnTimer = 0;
    fpSpawnGate();
  }

  const speed = 0.33;
  fp.gates.forEach((g) => {
    g.z -= speed * dt;
    if (!g.resolved && g.z <= 0.15) {
      const steady = fp.tremorSmooth !== null && fp.tremorSmooth <= fp.threshold;
      g.resolved = true;
      if (steady) {
        fp.passed += 1;
        fp.score += 10;
        fp.passStreak = Math.min(fp.passStreak + 1, 10);
        fp.failStreak = Math.max(fp.failStreak - 1, 0);
      } else {
        fp.missed += 1;
        fp.score = Math.max(0, fp.score - 6);
        fp.health = Math.max(0, fp.health - 10);
        fp.failStreak = Math.min(fp.failStreak + 1, 6);
        fp.passStreak = 0;
      }
    }
  });
  fp.gates = fp.gates.filter((g) => g.z > 0.02);

  const els = fpEls();
  if (els.score) els.score.textContent = String(fp.score);
  if (els.passed) els.passed.textContent = String(fp.passed);
  if (els.missed) els.missed.textContent = String(fp.missed);
  if (els.health) els.health.textContent = String(fp.health);
  if (els.status) {
    if (fp.health <= 0) els.status.textContent = 'Run ended — health depleted';
    else if (fp.tremorSmooth === null) els.status.textContent = 'Show hand to start gating';
    else els.status.textContent = (fp.tremorSmooth <= fp.threshold) ? 'Gate open' : 'Gate blocked — steady up';
  }

  fpDraw();
  if (fp.health <= 0) {
    fp.running = false;
    return;
  }
  fp.rafId = requestAnimationFrame(fpTick);
}

function fpStartStop() {
  const els = fpEls();
  if (!fp.running) {
    fp.running = true;
    fp.lastTs = 0;
    if (els.start) els.start.textContent = 'Pause';
    fp.rafId = requestAnimationFrame(fpTick);
  } else {
    fp.running = false;
    if (els.start) els.start.textContent = 'Resume';
    if (fp.rafId) cancelAnimationFrame(fp.rafId);
  }
}

function fpOnMetrics(m) {
  if (!fp.active) return;
  if (!metricsTracked(m)) {
    fp.tremorSmooth = null;
    return;
  }
  const score = parseFloat(m.live_score);
  const band = gateParsePercent(m.band_ratio);
  const amp = gateParseNum(m.amp_mm);
  if (!isNaN(score)) {
    fp.tremorSmooth = fp.tremorSmooth === null ? score : (0.3 * score + 0.7 * fp.tremorSmooth);
    fp.baseline.update(amp, band, score);
    const thr = fpRecomputeThreshold();
    const els = fpEls();
    if (els.threshold) els.threshold.textContent = 'Threshold: ' + thr.toFixed(0);
    if (els.live) els.live.textContent = 'Live tremor: ' + fp.tremorSmooth.toFixed(0);
  }
}

function enterFpGate() {
  fp.active = true;
  const els = fpEls();
  if (els.start && !els.start._wired) {
    els.start._wired = true;
    els.start.addEventListener('click', fpStartStop);
  }
  if (els.reset && !els.reset._wired) {
    els.reset._wired = true;
    els.reset.addEventListener('click', fpReset);
  }
  fpReset();
  fpDraw();
}

function leaveFpGate() {
  fp.active = false;
  fp.running = false;
  if (fp.rafId) cancelAnimationFrame(fp.rafId);
  const els = fpEls();
  if (els.start) els.start.textContent = 'Start run';
}

// ============================================================
// GAMES CATALOG — 7 embedded games (1 original + 6 from GitHub)
// ============================================================
const GAMES_CATALOG = [
  { name: 'Tremor Gauntlet',
    type: 'First-person raycaster shooter',
    use: 'WASD + mouse. 8 enemies. Weapon sway scales with your live tremor score.',
    url: 'mb-local://mb-fps/index.html',
    source: null, sourceUrl: null },
  { name: 'Andromeda Invaders',
    type: 'Space Invaders arcade',
    use: 'Arrow keys or A/D to move, Space to fire. Shoot alien ships before they descend. Includes synth music.',
    url: 'mb-local://gh-invaders/index.html',
    source: 'susam/invaders', sourceUrl: 'https://github.com/susam/invaders' },
  { name: 'Tetris',
    type: 'Block stacking puzzle',
    use: 'Arrow keys to move and rotate pieces. Clear complete rows before the stack reaches the top.',
    url: 'mb-local://gh-tetris/index.html',
    source: 'jakesgordon/javascript-tetris', sourceUrl: 'https://github.com/jakesgordon/javascript-tetris' },
  { name: 'Breakout',
    type: 'Brick breaker',
    use: 'Move mouse or left/right arrow keys to steer the paddle. Break all the bricks to win. 3 lives.',
    url: 'mb-local://gh-breakout/index.html',
    source: 'end3r/Gamedev-Canvas-workshop', sourceUrl: 'https://github.com/end3r/Gamedev-Canvas-workshop' },
  { name: 'Whack-A-Mole',
    type: 'Reaction speed game',
    use: 'Click the moles as they pop up from their holes. React fast to score points before the timer ends.',
    url: 'mb-local://gh-whackamole/index.html',
    source: 'Metroxe/one-html-page-challenge', sourceUrl: 'https://github.com/Metroxe/one-html-page-challenge' },
  { name: 'Game of Life',
    type: "Conway's cellular simulation",
    use: "Click cells to toggle them alive or dead, then press Start. Watch the colony evolve by Conway's rules.",
    url: 'mb-local://gh-gameoflife/index.html',
    source: 'Metroxe/one-html-page-challenge', sourceUrl: 'https://github.com/Metroxe/one-html-page-challenge' },
  { name: 'Hangman',
    type: 'Word guessing',
    use: 'Click letters or type on keyboard to guess the hidden word before the figure is complete. 9 wrong guesses allowed.',
    url: 'mb-local://gh-hangman/index.html',
    source: 'Metroxe/one-html-page-challenge', sourceUrl: 'https://github.com/Metroxe/one-html-page-challenge' },
];

function renderGames() {
  const grid = document.getElementById('games-grid');
  if (!grid) return;
  if (grid._rendered) return;
  grid._rendered = true;

  grid.innerHTML = GAMES_CATALOG.map((g, i) => {
    const sourceTag = g.source
      ? '<div class="game-card-source"><span class="game-card-source-icon">&#9670;</span>' +
        '<span class="game-card-source-label">GitHub: ' + g.source + '</span></div>'
      : '<div class="game-card-source"><span class="game-card-source-label">MotionBloom Original</span></div>';
    return '<div class="game-card" data-idx="' + i + '">' +
      '<div class="game-card-name">' + g.name + '</div>' +
      '<div class="game-card-type">' + g.type + '</div>' +
      '<div class="game-card-use">' + g.use + '</div>' +
      sourceTag +
      '<button class="game-card-launch local" data-url="' + g.url + '" data-name="' + g.name + '" data-local="true">' +
      'Play Now</button>' +
      '</div>';
  }).join('');

  grid.addEventListener('click', function (e) {
    const btn = e.target.closest('.game-card-launch');
    if (btn) {
      e.stopPropagation();
      launchGame(btn.dataset.url, btn.dataset.name, true);
    }
  });

  // Back button wires up here
  const backBtn = document.getElementById('game-iframe-back');
  if (backBtn && !backBtn._wired) {
    backBtn._wired = true;
    backBtn.addEventListener('click', closeGameIframe);
  }
}

function launchGame(url, name, isLocal) {
  if (!url) return;
  if (isLocal) {
    // Resolve mb-local:// → actual file path via bridge helper, or embed directly
    const localPath = url.replace('mb-local://', '');
    // Build file:// path by going up from renderer/ to electron/ then into games/
    const base = window.location.href.replace(/\/renderer\/[^/]+$/, '');
    const filePath = base + '/games/' + localPath;
    const iframeWrap = document.getElementById('game-iframe-wrap');
    const catalog    = document.getElementById('games-catalog');
    const iframe     = document.getElementById('game-iframe');
    const title      = document.getElementById('game-iframe-title');
    if (iframeWrap && catalog && iframe) {
      catalog.style.display    = 'none';
      iframeWrap.style.display = 'flex';
      if (title) title.textContent = name || 'Game';
      iframe.src = filePath;
    }
  } else {
    // External GitHub/web link — open in system browser
    if (window.motionbloomBridge && typeof window.motionbloomBridge.openExternal === 'function') {
      window.motionbloomBridge.openExternal(url);
    } else {
      window.open(url, '_blank');
    }
  }
}

function closeGameIframe() {
  const iframeWrap = document.getElementById('game-iframe-wrap');
  const catalog    = document.getElementById('games-catalog');
  const iframe     = document.getElementById('game-iframe');
  if (iframeWrap) iframeWrap.style.display = 'none';
  if (catalog)    catalog.style.display    = 'flex';
  if (iframe)     iframe.src               = 'about:blank';
}

// ---------- INIT ----------
setupCameraPreview();
showTab('home');

// Expose for later phases.
window.MotionBloomStore = Store;
window.MotionBloomRecordSession = recordSession;

// ============================================================
// SCHEDULE
// ============================================================
(function() {
  const SCHED_KEY = 'motionbloom.schedule.v1';

  function getEvents() {
    try { return JSON.parse(localStorage.getItem(SCHED_KEY) || '[]'); }
    catch(_e) { return []; }
  }
  function saveEvents(evs) {
    localStorage.setItem(SCHED_KEY, JSON.stringify(evs));
  }
  function addEvent(ev) {
    const evs = getEvents();
    evs.push(ev);
    saveEvents(evs);
  }
  function deleteEvent(id) {
    saveEvents(getEvents().filter(e => e.id !== id));
  }

  const TYPE_LABELS = { practice: 'Daily Practice', clinic: 'Clinic Visit', reminder: 'Reminder', custom: 'Custom' };
  const TYPE_COLORS = { practice: 'practice', clinic: 'clinic', reminder: 'reminder', custom: 'custom' };

  let calYear, calMonth;

  function isoDate(y, m, d) {
    return y + '-' + String(m+1).padStart(2,'0') + '-' + String(d).padStart(2,'0');
  }

  function renderSchedule() {
    const now = new Date();
    if (!calYear) { calYear = now.getFullYear(); calMonth = now.getMonth(); }
    renderCal();
    renderUpcoming();
    wireModal();
  }
  window._renderSchedule = renderSchedule;

  function renderCal() {
    const pane = document.getElementById('sched-cal-pane');
    if (!pane) return;
    const evs = getEvents();
    const now = new Date();
    const today = isoDate(now.getFullYear(), now.getMonth(), now.getDate());
    const firstDay = new Date(calYear, calMonth, 1).getDay();
    const daysInMonth = new Date(calYear, calMonth+1, 0).getDate();
    const monthName = new Date(calYear, calMonth, 1).toLocaleString('default', { month: 'long', year: 'numeric' });

    let html = '<div class="sched-header">' +
      '<button class="sched-nav" id="sched-prev">&lt;</button>' +
      '<span class="sched-month-title">' + monthName + '</span>' +
      '<button class="sched-nav" id="sched-next">&gt;</button>' +
      '</div><div class="cal-grid">';
    ['Su','Mo','Tu','We','Th','Fr','Sa'].forEach(d => { html += '<div class="cal-dow">' + d + '</div>'; });
    for (let i = 0; i < firstDay; i++) html += '<div class="cal-day empty"></div>';
    for (let d = 1; d <= daysInMonth; d++) {
      const dateStr = isoDate(calYear, calMonth, d);
      const dayEvs = evs.filter(e => e.date === dateStr);
      const isToday = dateStr === today;
      const isPast = dateStr < today;
      html += '<div class="cal-day' + (isToday?' today':'') + (isPast?' past':'') + '" data-date="' + dateStr + '">' +
        '<span class="cal-num">' + d + '</span>' +
        '<div class="cal-dots">' +
        dayEvs.slice(0,5).map(e => '<div class="cal-dot ' + (TYPE_COLORS[e.type]||'custom') + '"></div>').join('') +
        '</div></div>';
    }
    html += '</div>';
    pane.innerHTML = html;

    document.getElementById('sched-prev').addEventListener('click', () => {
      calMonth--; if(calMonth<0){calMonth=11;calYear--;} renderCal();
    });
    document.getElementById('sched-next').addEventListener('click', () => {
      calMonth++; if(calMonth>11){calMonth=0;calYear++;} renderCal();
    });
    pane.querySelectorAll('.cal-day:not(.empty)').forEach(cell => {
      cell.addEventListener('click', () => openModal(cell.dataset.date));
    });
  }

  function renderUpcoming() {
    const list = document.getElementById('sched-up-list');
    if (!list) return;
    const now = new Date();
    const todayStr = isoDate(now.getFullYear(), now.getMonth(), now.getDate());
    const evs = getEvents()
      .filter(e => e.date >= todayStr)
      .sort((a,b) => (a.date+a.time) < (b.date+b.time) ? -1 : 1)
      .slice(0, 20);
    if (!evs.length) { list.innerHTML = '<p class="sched-empty">No upcoming events.<br>Click any day to add one.</p>'; return; }
    list.innerHTML = evs.map(e => {
      const dObj = new Date(e.date + 'T' + (e.time||'00:00'));
      const dayLabel = e.date === todayStr ? 'Today' :
        dObj.toLocaleDateString('default', { month:'short', day:'numeric' });
      return '<div class="sched-ev-item">' +
        '<div class="sched-ev-dot ' + (TYPE_COLORS[e.type]||'custom') + '"></div>' +
        '<div class="sched-ev-info">' +
          '<div class="sched-ev-title">' + (TYPE_LABELS[e.type]||e.type) + '</div>' +
          '<div class="sched-ev-sub">' + dayLabel + (e.time ? ' · ' + e.time : '') + (e.note ? ' · ' + e.note : '') + '</div>' +
        '</div>' +
        '<button class="sched-ev-del" data-id="' + e.id + '" title="Delete">×</button>' +
        '</div>';
    }).join('');
    list.querySelectorAll('.sched-ev-del').forEach(btn => {
      btn.addEventListener('click', () => { deleteEvent(btn.dataset.id); renderUpcoming(); renderCal(); });
    });
  }

  function openModal(dateStr) {
    const bg = document.getElementById('sched-modal-bg');
    const label = document.getElementById('sched-modal-date-label');
    const existing = document.getElementById('sched-modal-existing');
    if (!bg) return;
    const d = new Date(dateStr + 'T00:00');
    if (label) label.textContent = 'Events — ' + d.toLocaleDateString('default', { weekday:'short', month:'short', day:'numeric' });
    // Show existing events for this day
    const dayEvs = getEvents().filter(e => e.date === dateStr);
    if (existing) {
      existing.innerHTML = dayEvs.map(e =>
        '<div class="sched-ev-item">' +
        '<div class="sched-ev-dot ' + (TYPE_COLORS[e.type]||'custom') + '"></div>' +
        '<div class="sched-ev-info"><div class="sched-ev-title">' + (TYPE_LABELS[e.type]||e.type) + '</div>' +
        '<div class="sched-ev-sub">' + (e.time||'') + (e.note?' · '+e.note:'') + '</div></div>' +
        '<button class="sched-ev-del" data-id="' + e.id + '">×</button>' +
        '</div>'
      ).join('');
      existing.querySelectorAll('.sched-ev-del').forEach(btn => {
        btn.addEventListener('click', () => { deleteEvent(btn.dataset.id); openModal(dateStr); renderUpcoming(); renderCal(); });
      });
    }
    bg.dataset.date = dateStr;
    bg.style.display = 'flex';
  }

  function wireModal() {
    const bg = document.getElementById('sched-modal-bg');
    const close = document.getElementById('sched-modal-close');
    const save = document.getElementById('sched-modal-save');
    const addBtn = document.getElementById('sched-add-today-btn');
    if (!bg) return;

    if (close && !close._wired) {
      close._wired = true;
      close.addEventListener('click', () => { bg.style.display='none'; });
    }
    bg.addEventListener('click', e => { if(e.target===bg) bg.style.display='none'; });

    if (save && !save._wired) {
      save._wired = true;
      save.addEventListener('click', () => {
        const type = document.getElementById('sched-type').value;
        const time = document.getElementById('sched-time').value;
        const note = document.getElementById('sched-note').value.trim();
        const repeat = document.getElementById('sched-repeat').value;
        const baseDate = bg.dataset.date;
        if (!baseDate) return;
        const dates = [baseDate];
        if (repeat === 'daily') {
          for (let i=1;i<30;i++) {
            const d=new Date(baseDate+'T00:00'); d.setDate(d.getDate()+i);
            dates.push(isoDate(d.getFullYear(),d.getMonth(),d.getDate()));
          }
        } else if (repeat === 'weekly') {
          for (let i=1;i<8;i++) {
            const d=new Date(baseDate+'T00:00'); d.setDate(d.getDate()+i*7);
            dates.push(isoDate(d.getFullYear(),d.getMonth(),d.getDate()));
          }
        }
        dates.forEach(dt => addEvent({ id: dt+'_'+type+'_'+Date.now(), date: dt, type, time, note }));
        bg.style.display = 'none';
        document.getElementById('sched-note').value = '';
        renderCal(); renderUpcoming();
      });
    }

    if (addBtn && !addBtn._wired) {
      addBtn._wired = true;
      addBtn.addEventListener('click', () => {
        const now = new Date();
        openModal(isoDate(now.getFullYear(), now.getMonth(), now.getDate()));
      });
    }
  }

  // Patch showTab to call our render
  const _origRenderSchedule = window._renderSchedule;
  window.renderSchedule = function() { if(_origRenderSchedule) _origRenderSchedule(); };
})();

// ============================================================
// REPORTS
// ============================================================
(function() {
  let reportRange = 7;

  function renderReports() {
    const rangeEl = document.getElementById('reports-range-btns');
    if (rangeEl && !rangeEl._wired) {
      rangeEl._wired = true;
      rangeEl.querySelectorAll('.rng-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          rangeEl.querySelectorAll('.rng-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          reportRange = parseInt(btn.dataset.range, 10);
          buildReports();
        });
      });
    }
    const expBtn = document.getElementById('reports-export-btn');
    if (expBtn && !expBtn._wired) {
      expBtn._wired = true;
      expBtn.addEventListener('click', exportReport);
    }
    buildReports();
  }
  window.renderReports = renderReports;

  function getFiltered() {
    const all = getSessions();
    if (!reportRange) return all;
    const cutoff = Date.now() - reportRange * 86400000;
    return all.filter(s => s.ts >= cutoff);
  }

  const DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];

  function buildReports() {
    const sessions = getFiltered();
    const allSessions = getSessions();
    const statRow = document.getElementById('reports-stat-row');
    const panels  = document.getElementById('reports-panels');
    if (!statRow || !panels) return;

    // ── Top stats ──────────────────────────────────────────────
    const avgScore = sessions.length
      ? Math.round(sessions.reduce((a,s) => a+s.score, 0) / sessions.length)
      : null;
    const bestScore = sessions.length ? Math.max(...sessions.map(s=>s.score)) : null;
    const totalSec  = sessions.reduce((a,s) => a+(s.dur||0), 0);
    const totalMin  = Math.round(totalSec/60);
    // Trend vs previous equal window
    let trendStr = '—';
    if (reportRange && sessions.length) {
      const cutoff = Date.now() - reportRange*86400000;
      const prev = allSessions.filter(s => s.ts >= cutoff - reportRange*86400000 && s.ts < cutoff);
      if (prev.length) {
        const prevAvg = prev.reduce((a,s)=>a+s.score,0)/prev.length;
        const diff = avgScore - prevAvg;
        trendStr = (diff >= 0 ? '▲' : '▼') + ' ' + Math.abs(Math.round(diff)) + ' pts vs prev period';
      }
    }

    statRow.innerHTML =
      rptStat(avgScore !== null ? avgScore : '—', 'Avg Score', trendStr) +
      rptStat(bestScore !== null ? bestScore : '—', 'Best Score', sessions.length + ' sessions') +
      rptStat(sessions.length, 'Sessions', (reportRange ? 'Last ' + reportRange + 'd' : 'All time')) +
      rptStat(totalMin + 'm', 'Time Practiced', totalSec > 0 ? formatDur(totalSec) : '—');

    // ── Per-exercise table ──────────────────────────────────────
    const exMap = {};
    sessions.forEach(s => {
      const ex = s.exercise || 'Unknown';
      if (!exMap[ex]) exMap[ex] = { scores:[], total:0 };
      exMap[ex].scores.push(s.score);
      exMap[ex].total++;
    });
    const exRows = Object.entries(exMap)
      .map(([ex, d]) => {
        const avg = Math.round(d.scores.reduce((a,b)=>a+b,0)/d.scores.length);
        const best = Math.max(...d.scores);
        const trend = d.scores.length > 1 ? (d.scores[d.scores.length-1] > d.scores[0] ? '▲' : '▼') : '—';
        return { ex, avg, best, count: d.total, trend };
      })
      .sort((a,b) => b.avg - a.avg);

    let exTable = '<table class="rpt-table"><thead><tr><th>Exercise</th><th>Avg</th><th>Best</th><th>Sessions</th><th>Trend</th></tr></thead><tbody>';
    if (exRows.length) {
      exRows.forEach(r => {
        exTable += '<tr><td>' + r.ex + '</td><td>' + r.avg + '</td><td>' + r.best + '</td><td>' + r.count + '</td><td class="rpt-trend">' + r.trend + '</td></tr>';
      });
    } else {
      exTable += '<tr><td colspan="5" style="color:var(--text-muted);text-align:center;padding:20px">No sessions yet.</td></tr>';
    }
    exTable += '</tbody></table>';

    // ── Day-of-week heatmap ─────────────────────────────────────
    const dowCounts = [0,0,0,0,0,0,0];
    const dowSums   = [0,0,0,0,0,0,0];
    sessions.forEach(s => {
      const d = new Date(s.ts).getDay();
      dowCounts[d]++;
      dowSums[d] += s.score;
    });
    const maxCount = Math.max(...dowCounts, 1);
    let heatHtml = '<div class="heatmap-label-row">' + DOW.map(d=>'<div class="heatmap-lbl">'+d+'</div>').join('') + '</div>';
    heatHtml += '<div class="heatmap-row">' + dowCounts.map((c,i) => {
      const alpha = 0.08 + (c/maxCount)*0.88;
      const avg = c ? Math.round(dowSums[i]/c) : null;
      return '<div class="heatmap-cell" style="background:rgba(224,30,55,' + alpha.toFixed(2) + ')">' +
        (c ? '<span>' + avg + '</span>' : '<span style="opacity:.3">—</span>') + '</div>';
    }).join('') + '</div>' +
    '<p style="font-size:10px;color:var(--text-muted);margin-top:7px">Colour = session count. Number = avg score on that day.</p>';

    // ── Insights ──────────────────────────────────────────────
    const insights = buildInsights(sessions, exRows, dowCounts, dowSums);

    // ── Assemble panels ────────────────────────────────────────
    panels.innerHTML =
      '<div class="rpt-panel full"><div class="rpt-panel-title">Score by Exercise</div>' + exTable + '</div>' +
      '<div class="rpt-panel"><div class="rpt-panel-title">Day of Week Heatmap</div>' + heatHtml + '</div>' +
      '<div class="rpt-panel"><div class="rpt-panel-title">Insights</div><div class="insight-list">' +
        insights.map(i => '<div class="insight-item"><span class="insight-bullet">' + i.icon + '</span><span>' + i.text + '</span></div>').join('') +
      '</div></div>';
  }

  function rptStat(val, label, sub) {
    return '<div class="rpt-stat"><div class="rpt-stat-val">' + val + '</div>' +
      '<div class="rpt-stat-label">' + label + '</div>' +
      '<div class="rpt-stat-sub">' + (sub||'&nbsp;') + '</div></div>';
  }

  function formatDur(sec) {
    if (sec < 60) return sec + 's';
    const h = (sec/3600)|0, m = ((sec%3600)/60)|0;
    return (h ? h + 'h ' : '') + m + 'm';
  }

  function buildInsights(sessions, exRows, dowCounts, dowSums) {
    const out = [];
    if (!sessions.length) {
      out.push({icon:'💡', text:'Complete your first session to see personalised insights here.'});
      return out;
    }
    // Best day
    const bestDow = dowCounts.reduce((best,c,i) => c>dowCounts[best]?i:best, 0);
    if (dowCounts[bestDow]) out.push({icon:'📅', text:'You practice most on <strong>' + DOW[bestDow] + 's</strong> (' + dowCounts[bestDow] + ' sessions).'});
    // Improvement trend
    if (sessions.length >= 6) {
      const first3 = sessions.slice(0,3).reduce((a,s)=>a+s.score,0)/3;
      const last3  = sessions.slice(-3).reduce((a,s)=>a+s.score,0)/3;
      const diff = Math.round(last3 - first3);
      if (diff > 0) out.push({icon:'📈', text:'Average score improved by <strong>+' + diff + ' pts</strong> compared to your first sessions.'});
      else if (diff < 0) out.push({icon:'📉', text:'Average score dropped by <strong>' + diff + ' pts</strong> vs your first sessions — keep going!'});
      else out.push({icon:'➡️', text:'Your score has been consistent — try pushing for a new personal best.'});
    }
    // Best exercise
    if (exRows.length) out.push({icon:'🏆', text:'Best exercise: <strong>' + exRows[0].ex + '</strong> (avg ' + exRows[0].avg + ').'});
    // Needs work
    if (exRows.length > 1) {
      const worst = exRows[exRows.length-1];
      out.push({icon:'🎯', text:'Focus area: <strong>' + worst.ex + '</strong> (avg ' + worst.avg + ') — room to improve.'});
    }
    // Consistency
    const streak = (function(){
      const days = new Set(sessions.map(s => new Date(s.ts).toDateString()));
      let max=0, cur=0, prev=null;
      const sorted=[...days].sort();
      sorted.forEach(d => {
        const dd = new Date(d);
        if(prev && (dd-prev)/86400000===1) cur++; else cur=1;
        if(cur>max) max=cur;
        prev=dd;
      });
      return max;
    })();
    if (streak >= 3) out.push({icon:'🔥', text:'Best streak: <strong>' + streak + ' consecutive days</strong> — great consistency!'});
    // Clinic reminder
    const schedEvs = JSON.parse(localStorage.getItem('motionbloom.schedule.v1') || '[]');
    const nextClinic = schedEvs.filter(e=>e.type==='clinic'&&e.date>=new Date().toISOString().slice(0,10)).sort((a,b)=>a.date<b.date?-1:1)[0];
    if (nextClinic) out.push({icon:'🏥', text:'Next clinic visit: <strong>' + nextClinic.date + '</strong>' + (nextClinic.note?' — '+nextClinic.note:'') + '.'});
    else out.push({icon:'🏥', text:'No clinic visit scheduled. Use the <strong>Schedule</strong> tab to add one.'});
    return out;
  }

  function exportReport() {
    const sessions = getSessions();
    const filtered = getFiltered();
    const title = 'MotionBloom Report — ' + new Date().toLocaleDateString();
    const avgScore = filtered.length ? Math.round(filtered.reduce((a,s)=>a+s.score,0)/filtered.length) : 'N/A';
    const best = filtered.length ? Math.max(...filtered.map(s=>s.score)) : 'N/A';
    const tableRows = filtered.slice().reverse().map(s =>
      '<tr><td>' + new Date(s.ts).toLocaleString() + '</td><td>' + (s.exercise||'—') + '</td><td>' + s.score + '</td><td>' + (s.dur ? formatDur(s.dur) : '—') + '</td></tr>'
    ).join('');
    const html = '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>' + title + '</title>' +
      '<style>body{font-family:system-ui,sans-serif;max-width:860px;margin:40px auto;color:#111;line-height:1.5}' +
      'h1{color:#e01e37}h2{margin-top:28px;font-size:15px;color:#555;text-transform:uppercase;letter-spacing:.06em}' +
      '.stats{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:20px}' +
      '.stat{background:#f8f9fa;border-radius:12px;padding:16px 22px}' +
      '.stat-val{font-size:32px;font-weight:900;color:#111}.stat-lbl{font-size:11px;color:#888;text-transform:uppercase}' +
      'table{width:100%;border-collapse:collapse}th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #eee}th{background:#f1f3f5;font-size:12px}</style></head>' +
      '<body><h1>🌸 MotionBloom Progress Report</h1>' +
      '<p>Generated: ' + new Date().toLocaleString() + ' &nbsp;·&nbsp; Period: ' + (reportRange ? 'Last ' + reportRange + ' days' : 'All time') + '</p>' +
      '<div class="stats">' +
      '<div class="stat"><div class="stat-val">' + avgScore + '</div><div class="stat-lbl">Avg Score</div></div>' +
      '<div class="stat"><div class="stat-val">' + best + '</div><div class="stat-lbl">Best Score</div></div>' +
      '<div class="stat"><div class="stat-val">' + filtered.length + '</div><div class="stat-lbl">Sessions</div></div>' +
      '<div class="stat"><div class="stat-val">' + Math.round(filtered.reduce((a,s)=>a+(s.dur||0),0)/60) + 'm</div><div class="stat-lbl">Time Practiced</div></div>' +
      '</div>' +
      '<h2>Session Log</h2>' +
      '<table><thead><tr><th>Date &amp; Time</th><th>Exercise</th><th>Score</th><th>Duration</th></tr></thead><tbody>' +
      tableRows + '</tbody></table>' +
      '</body></html>';

    // Try native save dialog via IPC, fallback to blob download
    if (window.motionbloomBridge && typeof window.motionbloomBridge.saveReport === 'function') {
      window.motionbloomBridge.saveReport(html, title + '.html');
    } else {
      const blob = new Blob([html], { type: 'text/html' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'MotionBloom_Report_' + new Date().toISOString().slice(0,10) + '.html';
      a.click();
    }
  }
})();
