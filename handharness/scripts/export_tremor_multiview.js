const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

function parseArg(name, fallback) {
  const token = `--${name}=`;
  const hit = process.argv.find((arg) => arg.startsWith(token));
  return hit ? hit.slice(token.length) : fallback;
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function buildCameraRig(distance) {
  const rig = [];

  // Tier 1: equatorial ring (16 cameras, every 22.5 degrees).
  for (let i = 0; i < 16; i += 1) {
    const yawDeg = i * 22.5;
    rig.push({
      id: `eq_${String(i + 1).padStart(2, "0")}`,
      tier: "equatorial",
      yawDeg,
      pitchDeg: 0,
      distance,
    });
  }

  // Tier 2: upper ring (8 cameras, every 45 degrees, looking downward).
  for (let i = 0; i < 8; i += 1) {
    const yawDeg = i * 45;
    rig.push({
      id: `up_${String(i + 1).padStart(2, "0")}`,
      tier: "upper",
      yawDeg,
      pitchDeg: 45,
      distance,
    });
  }

  // Tier 3: lower ring (8 cameras, every 45 degrees, looking upward).
  for (let i = 0; i < 8; i += 1) {
    const yawDeg = i * 45;
    rig.push({
      id: `low_${String(i + 1).padStart(2, "0")}`,
      tier: "lower",
      yawDeg,
      pitchDeg: -45,
      distance,
    });
  }

  return rig;
}

function buildTremorState(step, fps, hz, amp) {
  const t = step / fps;
  const p = 2 * Math.PI * hz * t;

  const phase = {
    thumb: 0.2,
    index: 1.1,
    middle: 1.9,
    ring: 2.7,
    pinky: 3.4,
  };

  const baseTwoUp = {
    thumb: { curl: 0.96, spread: -0.30, roll: 0.0 },
    index: { curl: 0.02, spread: -0.10, roll: 0.0 },
    middle: { curl: 0.02, spread: 0.10, roll: 0.0 },
    ring: { curl: 1.05, spread: 0.07, roll: 0.0 },
    pinky: { curl: 1.08, spread: 0.14, roll: 0.0 },
  };

  const osc = (w, extra = 0) => Math.sin(p + extra) * w;

  return {
    global: {
      rx: osc(amp * 0.10),
      ry: osc(amp * 0.06, Math.PI / 4),
      rz: osc(amp * 0.12, Math.PI / 7),
      tx: osc(amp * 0.006, Math.PI / 3),
      ty: osc(amp * 0.010, Math.PI / 5),
      tz: osc(amp * 0.005, Math.PI / 2),
      scale: 1,
    },
    wrist: {
      curl: osc(amp * 0.14, Math.PI / 6),
      yaw: osc(amp * 0.10, Math.PI / 2),
      roll: osc(amp * 0.15, Math.PI / 9),
    },
    palm: {
      arch: osc(amp * 0.08, Math.PI / 3),
      twist: osc(amp * 0.08, Math.PI / 8),
    },
    fingers: {
      thumb: {
        curl: baseTwoUp.thumb.curl + osc(amp * 0.09, phase.thumb),
        spread: baseTwoUp.thumb.spread + osc(amp * 0.04, phase.thumb + 0.3),
        roll: osc(amp * 0.05, phase.thumb + 0.6),
      },
      index: {
        curl: baseTwoUp.index.curl + osc(amp * 0.10, phase.index),
        spread: baseTwoUp.index.spread + osc(amp * 0.05, phase.index + 0.2),
        roll: osc(amp * 0.05, phase.index + 0.6),
      },
      middle: {
        curl: baseTwoUp.middle.curl + osc(amp * 0.10, phase.middle),
        spread: baseTwoUp.middle.spread + osc(amp * 0.05, phase.middle + 0.2),
        roll: osc(amp * 0.05, phase.middle + 0.6),
      },
      ring: {
        curl: baseTwoUp.ring.curl + osc(amp * 0.08, phase.ring),
        spread: baseTwoUp.ring.spread + osc(amp * 0.03, phase.ring + 0.2),
        roll: osc(amp * 0.05, phase.ring + 0.6),
      },
      pinky: {
        curl: baseTwoUp.pinky.curl + osc(amp * 0.08, phase.pinky),
        spread: baseTwoUp.pinky.spread + osc(amp * 0.03, phase.pinky + 0.2),
        roll: osc(amp * 0.05, phase.pinky + 0.6),
      },
    },
    _preset: "twoUp",
  };
}

async function launchBrowser() {
  const fallbackExecutable = path.join(
    process.env.HOME || "",
    "Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
  );

  const launchOptions = fs.existsSync(fallbackExecutable)
    ? { headless: true, executablePath: fallbackExecutable }
    : { headless: true };

  return chromium.launch(launchOptions);
}

async function main() {
  const url = parseArg("url", "http://127.0.0.1:4173/output/model/mesh_hand_viewer.html?v=tremor&capture=1");
  const outDir = path.resolve(parseArg("out", "output/tremor_multiview"));
  const steps = Number(parseArg("steps", "120"));
  const fps = Number(parseArg("fps", "60"));
  const hz = Number(parseArg("hz", "7.5"));
  const amp = Number(parseArg("amp", "1.0"));
  const distance = Number(parseArg("distance", "2.75"));
  const width = Number(parseArg("width", "768"));
  const height = Number(parseArg("height", "768"));

  const cameraRig = buildCameraRig(distance);
  ensureDir(outDir);

  const metadata = {
    type: "virtual_hand_tremor_multiview",
    captureMethod: "freeze_and_orbit",
    steps,
    fps,
    hz,
    amp,
    resolution: { width, height },
    cameraRig,
    notes: [
      "Each step is one tremor state; all cameras are captured at that same state.",
      "Equivalent to synchronized multi-camera capture with deterministic freeze-and-orbit.",
    ],
  };
  fs.writeFileSync(path.join(outDir, "capture_config.json"), JSON.stringify(metadata, null, 2));

  const browser = await launchBrowser();
  const page = await browser.newPage({ viewport: { width, height } });
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => !!window.handHarness, null, { timeout: 30000 });

  await page.evaluate(() => {
    window.handHarness.showHarness(false);
    window.handHarness.setHandMeshVisible(true);
    if (window.handHarness.setMeshDeformEnabled) {
      window.handHarness.setMeshDeformEnabled(true);
    }
    window.handHarness.posePreset("twoUp");
  });

  for (let step = 0; step < steps; step += 1) {
    const stateDir = path.join(outDir, `state_${String(step).padStart(5, "0")}`);
    ensureDir(stateDir);

    const tremorState = buildTremorState(step, fps, hz, amp);

    await page.evaluate((pose) => {
      window.handHarness.setPose(pose);
    }, tremorState);

    for (const cam of cameraRig) {
      await page.evaluate((view) => {
        window.handHarness.setCameraOrbit({
          yawDeg: view.yawDeg,
          pitchDeg: view.pitchDeg,
          distance: view.distance,
          targetX: 0,
          targetY: 0,
          targetZ: 0,
        });
      }, cam);

      const file = `${cam.id}.png`;
      await page.screenshot({ path: path.join(stateDir, file) });
    }

    fs.writeFileSync(
      path.join(stateDir, "pose_state.json"),
      JSON.stringify({ step, tremorState }, null, 2),
    );
  }

  await browser.close();
  console.log(`Tremor multiview export complete: ${outDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
