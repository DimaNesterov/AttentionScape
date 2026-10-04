"""
AttentionScape — геометрия: пол, лучи из камеры, пересечения.

Все координаты — в мировой системе W (метры, +Y вверх), см. docs/data_spec.md.
"""
import numpy as np

from .io import Frame, Mesh

FLOOR_CLASS = 2


def floor_height(mesh: Mesh) -> float:
    """Высота пола (Y) по треугольникам класса floor; запасной вариант — нижние 2% вершин."""
    floor = mesh.classes == FLOOR_CLASS
    if floor.sum() >= 50:
        vidx = np.unique(mesh.faces[floor].reshape(-1))
        return float(np.median(mesh.vertices[vidx, 1]))
    return float(np.percentile(mesh.vertices[:, 1], 2))


def pixel_ray(frame: Frame, u: float, v: float):
    """Луч из камеры через пиксель (u, v): начало и единичное направление в W."""
    T = frame.T_world_from_cam_cv
    d_cam = np.linalg.inv(frame.K) @ np.array([u, v, 1.0])
    d_world = T[:3, :3] @ d_cam
    return T[:3, 3].copy(), d_world / np.linalg.norm(d_world)


def intersect_horizontal(origin, direction, y: float):
    """Пересечение луча с горизонтальной плоскостью Y = y (только впереди камеры)."""
    if abs(direction[1]) < 1e-6:
        return None
    t = (y - origin[1]) / direction[1]
    return origin + t * direction if t > 0 else None


def closest_on_ray_to_vertical(origin, direction, point):
    """Точка луча, ближайшая (в плане XZ) к вертикали через point."""
    dxz = direction[[0, 2]]
    den = float(dxz @ dxz)
    if den < 1e-9:
        return None
    t = float((point - origin)[[0, 2]] @ dxz) / den
    return origin + t * direction if t > 0 else None
