# MotionBloom FPS Adaptation Backlog

This backlog turns the archived game list into immediate implementation work.

## Goal
Ship a playable first-person game mode with MotionBloom overlays and tremor-gate logic, then scale to stronger engines.

## Integration Contract (must exist in every game integration)
- Live metrics input:
  - `motion_score`
  - `tremor_score`
  - `dominant_frequency_hz`
  - `tracking_valid`
- Session logging:
  - per-second metrics stream
  - event logs for gate pass/fail
  - summary JSON per run
- UI overlay:
  - status (`TRACKING / LOST`)
  - threshold
  - current score
  - pass/fail events

## Priority Ladder

### P0 (start here)
1. `DOOM-3D-FPS-Shooting-Game` (Python/Pygame)
   - Fastest code-level adaptation.
   - Low engine complexity.
   - Best for proving overlay + gating loop.

2. `Fate` (browser retro FPS)
   - Fast web integration for demos.
   - Aligns with existing Electron path.

### P1 (next wave)
3. `AssaultCube`
4. `Cube 2: Sauerbraten`
5. `godot-3d-horror-game`
6. `Project Desolation`

### P2 (demo/hype or heavy lift)
7. `Red Eclipse`
8. `Xonotic`
9. `OpenTournament`
10. `SeriousSamClassic`

## Implementation Milestones

### M1 — Adapter Skeleton (in MotionBloom Electron)
- Add adapter interface for external game feeds:
  - `connect()`
  - `disconnect()`
  - `pushMetrics(metrics)`
  - `onGameEvent(callback)`
- Keep current in-app FP mode as baseline fallback.

### M2 — Overlay Runtime
- Build reusable HUD overlay package for:
  - score chips
  - tremor threshold gauge
  - pass/fail gate banner
- Plug into current `FP Tremor Gates` tab first.

### M3 — Gate Profiles
- Add gate profile presets:
  - `clinical_easy`
  - `clinical_standard`
  - `challenge`
- Each preset controls threshold and fail tolerance.

### M4 — Logging + Exports
- Persist session traces to:
  - `sessions/<timestamp>/metrics.jsonl`
  - `sessions/<timestamp>/summary.json`
- Include gate outcomes and confidence windows.

### M5 — First External Game Pilot
- Integrate one P0 repo end-to-end.
- Demo objective:
  - Start game
  - show live MotionBloom HUD
  - gates trigger correctly
  - export session report

## Acceptance Criteria for “Playable + Clinical”
- Stable 60s session without adapter crash.
- Overlay updates at least 10 Hz.
- Gate decisions explainable from saved metrics.
- `tracking_valid=false` handled gracefully (no hard crash, clear UI state).

## Immediate Next Build Target
- Use existing Electron implementation and ship **M1 + M2** first.
- Then pilot with `Fate` (web) or `DOOM-3D-FPS-Shooting-Game` (Python), depending on demo venue.
