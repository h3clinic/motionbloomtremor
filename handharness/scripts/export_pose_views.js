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

async function main() {
  const url = parseArg("url", "http://127.0.0.1:4173/output/model/mesh_hand_viewer.html?v=export&capture=1");
  const outDir = path.resolve(parseArg("out", "output/pose_renders"));
  const poseArg = parseArg("poses", "open,twoUp,fist,pinch,point,spread");
  const poses = poseArg.split(",").map((p) => p.trim()).filter(Boolean);

  const views = [
    { name: "front", yawDeg: 0, pitchDeg: 10, distance: 2.6 },
    { name: "front_left", yawDeg: 25, pitchDeg: 14, distance: 2.6 },
    { name: "left", yawDeg: 60, pitchDeg: 16, distance: 2.6 },
    { name: "back_left", yawDeg: 120, pitchDeg: 16, distance: 2.7 },
    { name: "back", yawDeg: 180, pitchDeg: 12, distance: 2.7 },
    { name: "back_right", yawDeg: 240, pitchDeg: 16, distance: 2.7 },
    { name: "right", yawDeg: 300, pitchDeg: 16, distance: 2.6 },
    { name: "front_right", yawDeg: 335, pitchDeg: 14, distance: 2.6 },
    { name: "top_front", yawDeg: 20, pitchDeg: 42, distance: 2.9 },
    { name: "top_back", yawDeg: 200, pitchDeg: 42, distance: 2.9 },
    { name: "low_front", yawDeg: 10, pitchDeg: -8, distance: 2.5 },
    { name: "low_right", yawDeg: 300, pitchDeg: -8, distance: 2.5 },
  ];

  ensureDir(outDir);

  const fallbackExecutable = path.join(
    process.env.HOME || "",
    "Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
  );

  const launchOptions = fs.existsSync(fallbackExecutable)
    ? { headless: true, executablePath: fallbackExecutable }
    : { headless: true };

  const browser = await chromium.launch(launchOptions);
  const page = await browser.newPage({ viewport: { width: 1280, height: 1280 } });
  await page.goto(url, { waitUntil: "domcontentloaded" });

  await page.waitForFunction(() => !!window.handHarness, null, { timeout: 20000 });

  for (const poseName of poses) {
    const poseDir = path.join(outDir, poseName);
    ensureDir(poseDir);

    for (let i = 0; i < views.length; i += 1) {
      const v = views[i];
      await page.evaluate(({ poseName, view }) => {
        window.handHarness.applyPoseAndView(poseName, {
          yawDeg: view.yawDeg,
          pitchDeg: view.pitchDeg,
          distance: view.distance,
          targetX: 0,
          targetY: 0,
          targetZ: 0,
        });
      }, { poseName, view: v });

      await page.waitForTimeout(120);
      const filename = `${String(i + 1).padStart(2, "0")}_${v.name}.png`;
      await page.screenshot({ path: path.join(poseDir, filename) });
    }
  }

  await browser.close();
  console.log(`Export complete: ${outDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
