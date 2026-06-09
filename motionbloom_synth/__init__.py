"""MotionBloom Synthetic Tremor Dataset Pipeline.

Architecture:
    3D controllable hand → scripted poses → injected tremor
    → rendered videos → landmark extraction → score 1–100
    → train classifier/regressor

Components:
    configs/    — YAML configuration (poses, tremor profiles, cameras, render)
    generator/  — Blender rendering pipeline (tremor engine, rig, camera, labels)
    training/   — ML pipeline (landmarks, features, Model A + B)
    outputs/    — Generated videos, labels, features, trained models
"""
