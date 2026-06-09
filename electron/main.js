const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

let mainWindow = null;
let bridgeProcess = null;
let stdoutBuffer = "";
let frameCount = 0;

const repoRoot = path.resolve(__dirname, "..");
// TopoTremor backend location. Prefer the copy vendored INSIDE the repo so a
// fresh `git clone` is self-contained; fall back to the legacy sibling layout
// (topotremor beside motionbloomtremor) so existing local checkouts still work.
const appRoot = path.resolve(repoRoot, "..");
function resolveTopoDir() {
  if (process.env.MB_TOPO_DIR) {
    return process.env.MB_TOPO_DIR;
  }
  const candidates = [
    path.join(repoRoot, "topotremor"), // vendored in-repo (preferred)
    path.join(appRoot, "topotremor"),  // legacy sibling layout
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(path.join(candidate, "topo_bridge.py"))) {
      return candidate;
    }
  }
  // Default to the in-repo path; startBridge will surface a clear error if the
  // bridge script is missing.
  return candidates[0];
}
const topoDir = resolveTopoDir();

// TEMP DIAGNOSTIC: append bridge lifecycle to a file so we can see what happens
// when the Topology Test tab starts the Python bridge. Safe to remove later.
function logBridge(msg) {
  try {
    fs.appendFileSync("/tmp/mb_bridge.log", `[${new Date().toISOString()}] ${msg}\n`);
  } catch (_e) { /* ignore */ }
}

function sendToRenderer(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

function parseBridgeStdout(chunk) {
  stdoutBuffer += chunk.toString("utf8");
  const lines = stdoutBuffer.split("\n");
  stdoutBuffer = lines.pop() || "";

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    try {
      const event = JSON.parse(trimmed);
      if (event && event.type === "frame") {
        frameCount += 1;
        if (frameCount % 15 === 1) {
          logBridge(`forwarded frame #${frameCount} (points=${event.points})`);
        }
      } else if (event && event.type === "status") {
        logBridge(`status -> renderer: running=${event.running} "${event.message || ""}"`);
      }
      sendToRenderer("bridge:event", event);
    } catch (_err) {
      sendToRenderer("bridge:event", {
        type: "log",
        level: "debug",
        message: trimmed,
      });
    }
  }
}

function resolvePython() {
  // The 3D Hand State Landscape bridge REQUIRES MediaPipe, which only has
  // wheels for Python 3.11 here (installed in .venv311). Prefer that first.
  // Order: explicit override → .venv311 (MediaPipe) → .venv → repo venv → system.
  if (process.env.MB_PYTHON) {
    return process.env.MB_PYTHON;
  }
  const candidates = [
    path.join(repoRoot, ".venv311", "bin", "python"), // in-repo MediaPipe env (preferred)
    path.join(repoRoot, ".venv", "bin", "python"),    // in-repo legacy env
    path.join(appRoot, ".venv311", "bin", "python"),  // sibling MediaPipe-capable (Py 3.11)
    path.join(appRoot, ".venv", "bin", "python"),     // sibling legacy (may lack MediaPipe)
    path.join(repoRoot, "venv", "bin", "python"),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return "python3";
}

function startBridge() {
  if (bridgeProcess) {
    return { ok: true, message: "Bridge already running" };
  }

  const pythonCommand = resolvePython();
  frameCount = 0;
  logBridge(`startBridge: spawning "${pythonCommand}" topo_bridge.py (cwd=${topoDir})`);
  // Landmark-free TopoTremor bridge: OpenCV optical flow + persistent homology.
  // Run from the topotremor/ directory so it can import topotremor_mvp.
  bridgeProcess = spawn(pythonCommand, ["topo_bridge.py"], {
    cwd: topoDir,
    stdio: ["pipe", "pipe", "pipe"],
    // MB_TREMOR_DEBUG=1 → reveal raw tremor evidence (don't hard-zero on
    // non-VALID states) and relax early-stage dot thresholds while tuning.
    env: { ...process.env, PYTHONUNBUFFERED: "1", MB_TREMOR_DEBUG: "1" },
  });

  bridgeProcess.stdout.on("data", parseBridgeStdout);
  bridgeProcess.stderr.on("data", (chunk) => {
    const text = chunk.toString("utf8").trim();
    logBridge(`stderr: ${text.slice(0, 300)}`);
    sendToRenderer("bridge:event", {
      type: "log",
      level: "error",
      message: text,
    });
  });

  bridgeProcess.on("error", (err) => {
    logBridge(`SPAWN ERROR: ${err && err.message ? err.message : err}`);
    sendToRenderer("bridge:event", {
      type: "status",
      running: false,
      message: `Bridge failed to start: ${err && err.message ? err.message : err}`,
    });
    bridgeProcess = null;
  });

  bridgeProcess.on("close", (code, signal) => {
    logBridge(`close: code=${code} signal=${signal || "none"}`);
    sendToRenderer("bridge:event", {
      type: "status",
      running: false,
      message: `Bridge exited (code=${code}, signal=${signal || "none"})`,
    });
    bridgeProcess = null;
    stdoutBuffer = "";
  });

  sendToRenderer("bridge:event", {
    type: "status",
    running: true,
    message: "Bridge started",
  });
  return { ok: true, message: "Bridge started" };
}

function stopBridge() {
  if (!bridgeProcess) {
    return { ok: true, message: "Bridge already stopped" };
  }
  bridgeProcess.kill("SIGTERM");
  return { ok: true, message: "Bridge stop requested" };
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 860,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: "#0f1117",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));

  // TEMP DIAGNOSTIC: mirror the renderer console + crashes into the bridge log
  // so we can see the renderer's view of events without opening DevTools.
  mainWindow.webContents.on("console-message", (_e, level, message, line, sourceId) => {
    if (level >= 1 || /enterTest|bridge|frame|tracking/i.test(message)) {
      const src = (sourceId || "").split("/").pop();
      logBridge(`renderer[${level}]: ${message} (${src}:${line})`);
    }
  });
  mainWindow.webContents.on("render-process-gone", (_e, details) => {
    logBridge(`RENDERER GONE: ${JSON.stringify(details)}`);
  });
}

app.whenReady().then(() => {
  createWindow();

  ipcMain.handle("bridge:start", () => startBridge());
  ipcMain.handle("bridge:stop", () => stopBridge());

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  stopBridge();
  if (process.platform !== "darwin") {
    app.quit();
  }
});
