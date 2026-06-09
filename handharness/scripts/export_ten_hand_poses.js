const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function clearPngs(dirPath) {
  for (const name of fs.readdirSync(dirPath)) {
    if (name.toLowerCase().endsWith(".png")) {
      fs.unlinkSync(path.join(dirPath, name));
    }
  }
}

function renderGalleryHtml(poses) {
  const cards = poses
    .map((p) => {
      const file = `${p.name}.png`;
      return `
      <a class="card" href="${file}" target="_blank" rel="noreferrer">
        <img src="${file}" alt="${p.name}" loading="lazy" />
        <span>${p.name}</span>
      </a>`;
    })
    .join("\n");

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>10 Hand Poses</title>
  <style>
    :root { color-scheme: dark; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background: radial-gradient(1200px 700px at 15% 10%, #173970 0%, #0a1633 50%, #060b1c 100%);
      color: #e8efff;
    }
    .wrap { max-width: 1300px; margin: 24px auto; padding: 0 16px; }
    h1 { margin: 0 0 8px; font-size: clamp(26px, 4vw, 56px); }
    p { margin: 0 0 20px; color: #c6d4f3; font-size: 18px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 14px;
    }
    .card {
      display: block;
      border: 1px solid rgba(170, 194, 240, 0.35);
      border-radius: 18px;
      overflow: hidden;
      text-decoration: none;
      background: rgba(7, 16, 40, 0.7);
      box-shadow: 0 14px 42px rgba(0, 0, 0, 0.28);
    }
    .card img { width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; }
    .card span {
      display: block;
      padding: 10px 12px;
      color: #dce7ff;
      font-weight: 600;
      letter-spacing: 0.02em;
    }
  </style>
</head>
<body>
  <main class="wrap">
    <h1>10 Different Hand Poses</h1>
    <p>Generated from the same 3D hand model in a separate window.</p>
    <section class="grid">${cards}
    </section>
  </main>
</body>
</html>`;
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
  const outDir = path.resolve("output/pose_samples_10");
  ensureDir(outDir);
  clearPngs(outDir);

  const poses = [
    {
      name: "01_open",
      custom: {
        fingers: {
          thumb: { curl: 0.02, spread: -0.10, roll: 0.0 },
          index: { curl: 0.03, spread: -0.03, roll: 0.0 },
          middle: { curl: 0.02, spread: 0.0, roll: 0.0 },
          ring: { curl: 0.03, spread: 0.06, roll: 0.0 },
          pinky: { curl: 0.04, spread: 0.11, roll: 0.0 },
        },
      },
    },
    {
      name: "02_spread",
      custom: {
        fingers: {
          thumb: { curl: 0.08, spread: -0.26, roll: 0.0 },
          index: { curl: 0.08, spread: -0.16, roll: 0.0 },
          middle: { curl: 0.07, spread: 0.0, roll: 0.0 },
          ring: { curl: 0.09, spread: 0.14, roll: 0.0 },
          pinky: { curl: 0.12, spread: 0.22, roll: 0.0 },
        },
      },
    },
    {
      name: "03_point_soft",
      custom: {
        fingers: {
          thumb: { curl: 0.20, spread: -0.18, roll: 0.0 },
          index: { curl: 0.06, spread: -0.03, roll: 0.0 },
          middle: { curl: 0.12, spread: 0.0, roll: 0.0 },
          ring: { curl: 0.38, spread: 0.07, roll: 0.0 },
          pinky: { curl: 0.42, spread: 0.12, roll: 0.0 },
        },
        wrist: { curl: 0.0, yaw: 0.01, roll: 0.01 },
      },
    },
    {
      name: "04_pinch_soft",
      custom: {
        fingers: {
          thumb: { curl: 0.36, spread: -0.30, roll: 0.0 },
          index: { curl: 0.34, spread: -0.18, roll: 0.0 },
          middle: { curl: 0.12, spread: 0.03, roll: 0.0 },
          ring: { curl: 0.10, spread: 0.09, roll: 0.0 },
          pinky: { curl: 0.12, spread: 0.14, roll: 0.0 },
        },
      },
    },
    {
      name: "05_relaxed_curl",
      custom: {
        fingers: {
          thumb: { curl: 0.20, spread: -0.12, roll: 0.0 },
          index: { curl: 0.18, spread: -0.06, roll: 0.0 },
          middle: { curl: 0.16, spread: 0.0, roll: 0.0 },
          ring: { curl: 0.20, spread: 0.06, roll: 0.0 },
          pinky: { curl: 0.24, spread: 0.12, roll: 0.0 },
        },
        wrist: { curl: 0.02, yaw: 0.0, roll: 0.0 },
      },
    },
    {
      name: "06_thumb_out",
      custom: {
        fingers: {
          thumb: { curl: 0.08, spread: -0.24, roll: 0.0 },
          index: { curl: 0.12, spread: -0.03, roll: 0.0 },
          middle: { curl: 0.14, spread: 0.0, roll: 0.0 },
          ring: { curl: 0.18, spread: 0.06, roll: 0.0 },
          pinky: { curl: 0.22, spread: 0.11, roll: 0.0 },
        },
        wrist: { curl: 0.0, yaw: 0.02, roll: 0.02 },
      },
    },
    {
      name: "07_soft_hook",
      custom: {
        fingers: {
          thumb: { curl: 0.18, spread: -0.14, roll: 0.0 },
          index: { curl: 0.30, spread: -0.06, roll: 0.0 },
          middle: { curl: 0.28, spread: 0.0, roll: 0.0 },
          ring: { curl: 0.26, spread: 0.08, roll: 0.0 },
          pinky: { curl: 0.24, spread: 0.14, roll: 0.0 },
        },
        wrist: { curl: 0.04, yaw: 0.0, roll: 0.0 },
      },
    },
    {
      name: "08_wrist_left",
      custom: {
        fingers: {
          thumb: { curl: 0.12, spread: -0.16, roll: 0.0 },
          index: { curl: 0.14, spread: -0.05, roll: 0.0 },
          middle: { curl: 0.12, spread: 0.0, roll: 0.0 },
          ring: { curl: 0.16, spread: 0.06, roll: 0.0 },
          pinky: { curl: 0.18, spread: 0.11, roll: 0.0 },
        },
        wrist: { curl: 0.0, yaw: -0.12, roll: -0.04 },
      },
    },
    {
      name: "09_wrist_right",
      custom: {
        fingers: {
          thumb: { curl: 0.12, spread: -0.16, roll: 0.0 },
          index: { curl: 0.14, spread: -0.05, roll: 0.0 },
          middle: { curl: 0.12, spread: 0.0, roll: 0.0 },
          ring: { curl: 0.16, spread: 0.06, roll: 0.0 },
          pinky: { curl: 0.18, spread: 0.11, roll: 0.0 },
        },
        wrist: { curl: 0.0, yaw: 0.12, roll: 0.04 },
      },
    },
    {
      name: "10_palm_arch",
      custom: {
        fingers: {
          thumb: { curl: 0.16, spread: -0.18, roll: 0.0 },
          index: { curl: 0.18, spread: -0.05, roll: 0.0 },
          middle: { curl: 0.20, spread: 0.0, roll: 0.0 },
          ring: { curl: 0.22, spread: 0.07, roll: 0.0 },
          pinky: { curl: 0.24, spread: 0.13, roll: 0.0 },
        },
        palm: { arch: 0.08, twist: 0.06 },
        wrist: { curl: 0.02, yaw: 0.0, roll: 0.0 },
      },
    },
  ];

  const browser = await launchBrowser();
  const page = await browser.newPage({ viewport: { width: 1024, height: 1024 } });
  await page.goto("http://127.0.0.1:4173/output/model/mesh_hand_viewer.html?v=tenposes&capture=1", { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => !!window.handHarness, null, { timeout: 30000 });

  await page.evaluate(() => {
    window.handHarness.showHarness(false);
    window.handHarness.setHandMeshVisible(true);
    if (window.handHarness.setMeshDeformEnabled) {
      window.handHarness.setMeshDeformEnabled(true);
    }
  });

  const basePose = {
    global: { rx: 0, ry: 0, rz: 0, tx: 0, ty: 0, tz: 0, scale: 1 },
    wrist: { curl: 0, yaw: 0, roll: 0 },
    palm: { arch: 0, twist: 0 },
    fingers: {
      thumb: { curl: 0, spread: 0, roll: 0 },
      index: { curl: 0, spread: 0, roll: 0 },
      middle: { curl: 0, spread: 0, roll: 0 },
      ring: { curl: 0, spread: 0, roll: 0 },
      pinky: { curl: 0, spread: 0, roll: 0 },
    },
  };

  for (const pose of poses) {
    await page.evaluate(({ pose, basePose }) => {
      window.handHarness.setCameraOrbit({ yawDeg: 14, pitchDeg: 12, distance: 2.8, targetX: 0, targetY: 0, targetZ: 0 });
      if (pose.preset) {
        window.handHarness.posePreset(pose.preset);
        return;
      }
      const p = JSON.parse(JSON.stringify(basePose));
      if (pose.custom?.fingers) {
        Object.entries(pose.custom.fingers).forEach(([finger, vals]) => {
          Object.assign(p.fingers[finger], vals);
        });
      }
      if (pose.custom?.wrist) {
        Object.assign(p.wrist, pose.custom.wrist);
      }
      if (pose.custom?.palm) {
        Object.assign(p.palm, pose.custom.palm);
      }
      p._preset = "open";
      window.handHarness.setPose(p);
    }, { pose, basePose });

    await page.waitForTimeout(120);
    await page.screenshot({ path: path.join(outDir, `${pose.name}.png`) });
  }

  fs.writeFileSync(path.join(outDir, "poses.json"), JSON.stringify(poses, null, 2));
  fs.writeFileSync(path.join(outDir, "index.html"), renderGalleryHtml(poses));
  await browser.close();
  console.log(`10 hand poses exported: ${outDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
