# Storefront submission checklist

## Microsoft Store (Partner Center)

Path: direct MSIX upload — no rewrite required.

1. **Register** as an individual dev at https://partner.microsoft.com/dashboard (one-time ~$19 USD).
2. **Reserve app name** "MotionBloom MotionBloom" in Partner Center → Apps and games → New product → MSIX or PWA.
3. **Update** [AppxManifest.xml](AppxManifest.xml):
   - Replace `Identity Name`, `Publisher`, and `Version` with the exact values shown on the reserved-app page (`Product Identity` section). Publisher **must** match `CN=<yourPublisherId>`.
4. **Generate Store assets** (PNG, transparent):
   - `Assets/StoreLogo.png` — 50×50
   - `Assets/Square44x44Logo.png` — 44×44
   - `Assets/Square150x150Logo.png` — 150×150
   - `Assets/Wide310x150Logo.png` — 310×150
   Place under `packaging/Assets/` and commit.
5. **Tag a release** (`git tag v0.1.x && git push origin v0.1.x`). The workflow builds `MotionBloom-windows-x64.msix` and attaches it to the GitHub release.
6. **Upload** the `.msix` to Partner Center → Packages.
7. **Store listing** — fill in description, screenshots (min 1, 1366×768 or 1920×1080), age rating, and paste the contents of [../PRIVACY.md](../PRIVACY.md) into the Privacy Policy URL field (or host it at `https://h3clinic.github.io/motionbloomtremor/PRIVACY`).
8. **Submit** — automated cert + malware scan runs; typical approval ≈ 24–48 h.

## Apple notarization (direct download, not Mac App Store)

Path: ship a notarized `.app` so users don't see Gatekeeper warnings. No sandboxing or Xcode rewrite required.

1. Enroll: https://developer.apple.com/programme/ ($99 USD / year).
2. In **Xcode → Settings → Accounts** create a **Developer ID Application** certificate. Export it as `.p12` with a password.
3. Generate an **app-specific password** at https://appleid.apple.com → Sign-In and Security → App-Specific Passwords.
4. Add these GitHub repo secrets (Settings → Secrets and variables → Actions):
   | Secret | Value |
   | --- | --- |
   | `APPLE_ID` | Your Apple ID email |
   | `APPLE_TEAM_ID` | 10-char team ID from https://developer.apple.com/account |
   | `APPLE_APP_PASSWORD` | The app-specific password |
   | `APPLE_CERT_P12_BASE64` | `base64 -i DeveloperID.p12 \| pbcopy` |
   | `APPLE_CERT_PASSWORD` | The password you set when exporting |
   | `APPLE_SIGNING_IDENTITY` | e.g. `Developer ID Application: Your Name (TEAMID)` |
5. Tag a release. Workflow auto-signs, notarizes, and staples the `.app` before zipping.

## Mac App Store (defer — high effort)

Requires sandboxing + Xcode 26 SDK. Skip unless needed for credibility; direct-download notarized build covers 95% of users.
