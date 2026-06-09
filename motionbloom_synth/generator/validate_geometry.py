"""Geometry Validation.

Rejects invalid meshes before they enter the training pipeline.
Catches:
  - Exploded triangle meshes (fake MANO, broken exports)
  - NaN/Inf vertices
  - Disconnected components
  - Extreme aspect ratios (not hand-shaped)
  - Zero-area degenerate faces

A mesh MUST pass all checks before it can be used for rendering or training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def validate_obj(path: Path) -> Tuple[bool, List[str]]:
    """Validate an OBJ mesh file.

    Args:
        path: Path to .obj file.

    Returns:
        (is_valid, list_of_errors). Empty error list = valid.
    """
    verts, faces = _load_obj(path)
    return validate_mesh(verts, faces, source=str(path))


def validate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    source: str = "unknown",
) -> Tuple[bool, List[str]]:
    """Validate a mesh given vertices and faces.

    Args:
        vertices: (N, 3) array of vertex positions.
        faces: (M, 3) array of triangle indices.
        source: Label for error messages.

    Returns:
        (is_valid, list_of_errors).
    """
    errors = []

    # --- Check 1: NaN / Inf vertices ---
    if np.any(np.isnan(vertices)):
        errors.append(f"[{source}] Contains NaN vertices")
    if np.any(np.isinf(vertices)):
        errors.append(f"[{source}] Contains Inf vertices")

    if errors:
        return False, errors  # Can't proceed with other checks

    # --- Check 2: Minimum vertex/face count ---
    if len(vertices) < 100:
        errors.append(f"[{source}] Too few vertices ({len(vertices)}). Min 100 for a hand mesh.")
    if len(faces) < 50:
        errors.append(f"[{source}] Too few faces ({len(faces)}). Min 50 for a hand mesh.")

    # --- Check 3: Bounding box aspect ratio ---
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    extents = bbox_max - bbox_min
    extents_sorted = np.sort(extents)[::-1]  # Largest first

    if extents_sorted[0] < 1e-6:
        errors.append(f"[{source}] Degenerate mesh: zero bounding box extent")
    else:
        aspect_ratio = extents_sorted[0] / max(extents_sorted[2], 1e-8)
        # A hand has roughly 2:1 to 4:1 aspect ratio (length vs thickness)
        # Allow up to 10:1 for extreme wrist extensions, reject >15:1
        if aspect_ratio > 15.0:
            errors.append(
                f"[{source}] Extreme aspect ratio {aspect_ratio:.1f}:1. "
                f"Expected <15:1 for a hand mesh. Likely exploded geometry."
            )

    # --- Check 4: Face index bounds ---
    if len(faces) > 0:
        max_idx = faces.max()
        if max_idx >= len(vertices):
            errors.append(
                f"[{source}] Face indices out of bounds: max={max_idx}, "
                f"but only {len(vertices)} vertices."
            )

    # --- Check 5: Degenerate (zero-area) faces ---
    if len(faces) > 0 and len(vertices) > 0:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1) * 0.5
        degenerate_count = np.sum(areas < 1e-10)
        degenerate_fraction = degenerate_count / len(faces)
        if degenerate_fraction > 0.1:
            errors.append(
                f"[{source}] {degenerate_fraction:.0%} degenerate faces "
                f"({degenerate_count}/{len(faces)}). Mesh is likely invalid."
            )

    # --- Check 6: Connected components (triangle adjacency) ---
    if len(faces) > 0 and len(vertices) > 0:
        n_components = _count_connected_components(faces, len(vertices))
        if n_components > 5:
            errors.append(
                f"[{source}] {n_components} disconnected components. "
                f"A valid hand mesh should be 1-3 components (mesh + optional nails)."
            )

    # --- Check 7: Vertex explosion (standard deviation) ---
    centroid = vertices.mean(axis=0)
    distances = np.linalg.norm(vertices - centroid, axis=1)
    std_dist = distances.std()
    mean_dist = distances.mean()
    if mean_dist > 0 and std_dist / mean_dist > 2.0:
        errors.append(
            f"[{source}] Vertex distribution too spread: std/mean={std_dist/mean_dist:.2f}. "
            f"Likely exploded geometry."
        )

    return len(errors) == 0, errors


def _count_connected_components(faces: np.ndarray, n_verts: int) -> int:
    """Count connected components using union-find on face adjacency."""
    parent = list(range(n_verts))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for f in faces:
        union(f[0], f[1])
        union(f[1], f[2])

    # Count unique roots among vertices that appear in faces
    used_verts = set(faces.flatten())
    roots = set(find(v) for v in used_verts)
    return len(roots)


def _load_obj(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from OBJ file."""
    verts = []
    faces = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                # Handle f v1 v2 v3 or f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                indices = []
                for p in parts[1:4]:
                    idx = int(p.split("/")[0]) - 1
                    indices.append(idx)
                faces.append(indices)

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def validate_mano_pkl(pkl_path: Path) -> Tuple[bool, List[str]]:
    """Validate that a MANO .pkl file contains the real trained model.

    Checks structure, vertex count, face topology, and rejects
    synthetic/placeholder files.

    Args:
        pkl_path: Path to MANO_RIGHT.pkl or MANO_LEFT.pkl.

    Returns:
        (is_valid, list_of_errors).
    """
    import pickle

    errors = []

    if not pkl_path.exists():
        return False, [f"File not found: {pkl_path}"]

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except Exception as e:
        return False, [f"Cannot load pkl: {e}"]

    # Required keys in official MANO
    required_keys = ["v_template", "f", "weights", "J_regressor", "posedirs", "shapedirs"]
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: '{key}'")

    if errors:
        return False, errors

    # Official MANO has exactly 778 vertices and 1538 faces
    v_template = np.array(data["v_template"])
    faces = np.array(data["f"])

    if v_template.shape != (778, 3):
        errors.append(
            f"v_template shape {v_template.shape} != (778, 3). "
            f"This is not a valid MANO model."
        )

    if faces.shape != (1538, 3):
        errors.append(
            f"faces shape {faces.shape} != (1538, 3). "
            f"This is not a valid MANO model."
        )

    # Check template mesh is valid geometry (not random noise)
    if v_template.shape == (778, 3):
        is_valid_geo, geo_errors = validate_mesh(v_template, faces, source=str(pkl_path))
        if not is_valid_geo:
            errors.extend(geo_errors)
            errors.append("MANO template mesh failed geometry validation. File may be synthetic/fake.")

    # Check skinning weights are proper (sum to ~1 per vertex)
    weights = np.array(data["weights"])
    if weights.shape[0] == 778:
        weight_sums = weights.sum(axis=1)
        bad_weights = np.abs(weight_sums - 1.0) > 0.01
        if bad_weights.sum() > 50:
            errors.append(
                f"Skinning weights don't sum to 1.0 for {bad_weights.sum()}/778 vertices. "
                f"This is not a valid MANO model."
            )

    return len(errors) == 0, errors
