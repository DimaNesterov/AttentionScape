"""
AttentionScape — накопление карты внимания на меше с учётом перекрытий.

Для каждого замера взгляда комната «рендерится из глаз человека»: виртуальная
камера в голове смотрит по направлению взгляда, каждый пиксель знает, какой
треугольник в нём ближайший. Вклад треугольника = гауссиана от угла между
пикселем и центром взгляда (σ конуса) × телесный угол пикселя × вес замера.
Поверхности, закрытые другими предметами, вклада не получают.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .io import Mesh

UP = np.array([0.0, 1.0, 0.0])


def _look_basis(gaze):
    fwd = gaze / np.linalg.norm(gaze)
    ref = UP if abs(fwd @ UP) < 0.95 else np.array([1.0, 0.0, 0.0])
    right = np.cross(fwd, ref)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def render_face_ids(mesh: Mesh, eye, gaze, half_fov: float, size: int, centroids):
    """Рендер номеров треугольников из точки eye вдоль gaze. Возвращает (ids[size,size], фокус); −1 — пусто."""
    right, up, fwd = _look_basis(np.asarray(gaze, dtype=float))
    rel = mesh.vertices.astype(np.float64) - np.asarray(eye, dtype=float)
    x, y, z = rel @ right, rel @ up, rel @ fwd
    focal = (size / 2) / np.tan(half_fov)

    zf = z[mesh.faces]
    ok = (zf > 0.05).all(axis=1)
    crel = centroids - np.asarray(eye, dtype=float)
    cos_angle = (crel @ fwd) / np.maximum(np.linalg.norm(crel, axis=1), 1e-9)
    ok &= cos_angle > np.cos(min(half_fov + np.radians(15), np.radians(89)))   # отсекаем всё вне конуса
    sel = np.nonzero(ok)[0]
    ids = np.full((size, size), -1, dtype=np.int64)
    if len(sel) == 0:
        return ids, focal

    fz = zf[sel]
    u = focal * x[mesh.faces[sel]] / fz + size / 2
    v = -focal * y[mesh.faces[sel]] / fz + size / 2
    order = np.argsort(-fz.mean(axis=1))                  # дальние сначала, ближние поверх
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    us, vs = u[order].tolist(), v[order].tolist()
    for k, (uu, vv) in zip(sel[order].tolist(), zip(us, vs)):
        fid = k + 1
        draw.polygon(list(zip(uu, vv)), fill=(fid & 255, (fid >> 8) & 255, (fid >> 16) & 255))
    a = np.asarray(img).astype(np.int64)
    ids = a[..., 0] + (a[..., 1] << 8) + (a[..., 2] << 16) - 1
    return ids, focal


def _pixel_geometry(size: int, focal: float):
    c = np.arange(size) + 0.5 - size / 2
    xx, yy = np.meshgrid(c, c)
    theta = np.arctan(np.sqrt(xx ** 2 + yy ** 2) / focal)
    return theta, np.cos(theta) ** 3                      # угол от центра и телесный угол пикселя (отн.)


def accumulate(mesh: Mesh, observations, size: int = 128):
    """observations: список словарей с eye, gaze, sigma (рад), weight.
    Возвращает (внимание по треугольникам, покрытие по треугольникам, точки попадания центра взгляда)."""
    n_faces = len(mesh.faces)
    heat = np.zeros(n_faces)
    coverage = np.zeros(n_faces)
    centroids = mesh.vertices[mesh.faces].astype(np.float64).mean(axis=1)
    hits = []
    for ob in observations:
        half_fov = min(2.5 * ob["sigma"], np.radians(60))
        ids, focal = render_face_ids(mesh, ob["eye"], ob["gaze"], half_fov, size, centroids)
        theta, solid = _pixel_geometry(size, focal)
        m = ids >= 0
        weight = np.exp(-theta ** 2 / (2 * ob["sigma"] ** 2)) * solid
        heat += np.bincount(ids[m], weights=weight[m] * ob["weight"], minlength=n_faces)
        coverage += np.bincount(ids[m], weights=solid[m] * ob["weight"], minlength=n_faces)
        center_id = ids[size // 2, size // 2]
        hits.append(centroids[center_id].round(3).tolist() if center_id >= 0 else None)
    return heat, coverage, hits


def face_areas(mesh: Mesh):
    a, b, c = (mesh.vertices[mesh.faces[:, k]].astype(np.float64) for k in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def faces_to_vertices(mesh: Mesh, per_face):
    nv = len(mesh.vertices)
    flat = mesh.faces.reshape(-1)
    total = np.bincount(flat, weights=np.repeat(per_face, 3), minlength=nv)
    count = np.bincount(flat, minlength=nv)
    return total / np.maximum(count, 1)


def write_results(session_dir: Path, session_info: dict, mesh: Mesh, attention_v, coverage_v,
                  time_range, params: dict):
    """results/attention.json + attention_f32.bin по docs/data_spec.md (раздел 4.4), один временной интервал."""
    out = Path(session_dir) / "results"
    out.mkdir(exist_ok=True)
    meta = {
        "schema": "attentionscape/0.1",
        "session_id": session_info.get("session_id"),
        "mesh_sha256": session_info.get("mesh_sha256"),
        "n_vertices": int(len(mesh.vertices)),
        "n_time_bins": 1,
        "time_bin_edges": [float(time_range[0]), float(time_range[1])],
        "layout": "attention[t][v], затем coverage[v]",
        "normalization": "density_per_m2",
        "params": params,
    }
    (out / "attention.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    np.concatenate([np.asarray(attention_v, "<f4"), np.asarray(coverage_v, "<f4")]).tofile(out / "attention_f32.bin")
    return out