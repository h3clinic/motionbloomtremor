"""Blender Render Pipeline.

Orchestrates headless Blender rendering for synthetic tremor videos.
Sets up scene, materials, lighting, background, and renders to MP4.

Must be run inside Blender's Python environment OR spawned via:
    blender --background --python render_blender.py -- [args]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_render_config() -> dict:
    """Load render profiles configuration."""
    config_path = CONFIGS_DIR / "render_profiles.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_scene(
    resolution: int = 512,
    fps: int = 30,
    duration_sec: float = 4.0,
    background: str = "white_clean",
):
    """Configure the Blender scene for rendering.

    Args:
        resolution: Output resolution (square).
        fps: Frames per second.
        duration_sec: Video duration.
        background: Background preset name from config.
    """
    import bpy

    scene = bpy.context.scene
    config = load_render_config()

    # Resolution
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    # Frame range
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = int(duration_sec * fps)

    # Output format: FFmpeg / H.264
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"

    # Background color
    bg_config = config["backgrounds"].get(background, config["backgrounds"]["white_clean"])
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        color = bg_config["color"]
        bg_node.inputs[0].default_value = (color[0], color[1], color[2], 1.0)


def setup_lighting(lighting_preset: str = "indoor_soft"):
    """Set up scene lighting from preset.

    Args:
        lighting_preset: Key from render_profiles.yaml lighting section.
    """
    import bpy
    from mathutils import Vector

    config = load_render_config()
    light_config = config["lighting"].get(lighting_preset)
    if light_config is None:
        light_config = config["lighting"]["indoor_soft"]

    # Remove existing lights
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Create light
    light_type = light_config["type"].upper()
    if light_type == "AREA":
        light_data = bpy.data.lights.new("SynthLight", "AREA")
        light_data.size = light_config.get("size", 2.0)
    elif light_type == "POINT":
        light_data = bpy.data.lights.new("SynthLight", "POINT")
    elif light_type == "SUN":
        light_data = bpy.data.lights.new("SynthLight", "SUN")
    else:
        light_data = bpy.data.lights.new("SynthLight", "POINT")

    light_data.energy = light_config["energy"]
    color = light_config["color"]
    light_data.color = (color[0], color[1], color[2])

    light_obj = bpy.data.objects.new("SynthLightObj", light_data)
    bpy.context.scene.collection.objects.link(light_obj)

    # Position
    pos = light_config.get("position", [0, 0, 2])
    light_obj.location = Vector(pos)

    # Ambient approximation (adjust world strength)
    ambient = light_config.get("ambient", 0.3)
    if bpy.context.scene.world and bpy.context.scene.world.use_nodes:
        bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs[1].default_value = ambient


def setup_skin_material(mesh_object, skin_tone: str = "medium"):
    """Apply skin tone material to the hand mesh.

    Args:
        mesh_object: Blender mesh object.
        skin_tone: Key from render_profiles.yaml skin_tones.
    """
    import bpy

    config = load_render_config()
    tone_config = config["skin_tones"].get(skin_tone, config["skin_tones"]["medium"])

    # Create material
    mat = bpy.data.materials.get("SynthSkin")
    if mat is None:
        mat = bpy.data.materials.new("SynthSkin")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Get or create Principled BSDF
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    base_color = tone_config["base_color"]
    bsdf.inputs["Base Color"].default_value = (
        base_color[0], base_color[1], base_color[2], 1.0
    )
    bsdf.inputs["Subsurface Weight"].default_value = tone_config.get("subsurface", 0.3)
    bsdf.inputs["Roughness"].default_value = tone_config.get("roughness", 0.4)

    # Assign material to mesh
    if mesh_object.data.materials:
        mesh_object.data.materials[0] = mat
    else:
        mesh_object.data.materials.append(mat)


def render_video(output_path: Path):
    """Render the current scene animation to an MP4 file.

    Args:
        output_path: Full path for the output video file.
    """
    import bpy

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set output path (Blender appends frame range)
    scene = bpy.context.scene
    scene.render.filepath = str(output_path.with_suffix(""))

    # Use Eevee for speed (switch to Cycles for quality if needed)
    scene.render.engine = "BLENDER_EEVEE_NEXT"

    # Render animation
    bpy.ops.render.render(animation=True)


def apply_degradation(output_path: Path, degradation: str = "none"):
    """Apply post-processing degradation to rendered video.

    Uses ffmpeg for compression/blur/noise simulation.

    Args:
        output_path: Path to the rendered video.
        degradation: Degradation preset name.
    """
    import subprocess

    config = load_render_config()
    deg_config = config["degradation"].get(degradation)

    if deg_config is None or degradation == "none":
        return

    blur = deg_config.get("blur_radius", 0)
    noise = deg_config.get("noise_sigma", 0)
    quality = deg_config.get("compression_quality", 95)

    # Build ffmpeg filter chain
    filters = []
    if blur > 0:
        filters.append(f"boxblur={blur}:{blur}")
    if noise > 0:
        # Approximate noise with random grain
        filters.append(f"noise=alls={noise}:allf=t")

    # CRF from quality (lower quality = higher CRF)
    crf = max(15, int(51 * (1 - quality / 100)))

    temp_path = output_path.with_suffix(".tmp.mp4")
    output_path.rename(temp_path)

    cmd = ["ffmpeg", "-y", "-i", str(temp_path)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-crf", str(crf), "-c:v", "libx264", str(output_path)]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        temp_path.unlink()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If ffmpeg fails, keep original
        if temp_path.exists():
            temp_path.rename(output_path)
