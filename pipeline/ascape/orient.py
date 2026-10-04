"""
AttentionScape — разворот кадров «верхом вверх».

Кадры ARKit сохраняются в ландшафтной ориентации сенсора. Если телефон стоял
вертикально, комната на кадре лежит на боку, а модели людей и лиц обучены на
вертикальных изображениях. Поза камеры говорит, где на кадре верх (по гравитации),
поэтому кадр можно развернуть на k × 90° перед распознаванием, а результаты
пересчитать обратно в исходные пиксели и оси камеры.
"""
import numpy as np
from PIL import Image

from .io import Frame


def _rotate_dir(v, k: int):
    """Направление на кадре (x вправо, y вниз) после поворота изображения на k × 90° против часовой."""
    x, y = float(v[0]), float(v[1])
    for _ in range(k % 4):
        x, y = y, -x
    return np.array([x, y])


def upright_rotation(frame: Frame) -> int:
    """Сколько раз повернуть кадр на 90° против часовой, чтобы мировой «верх» смотрел вверх."""
    up_cam = frame.T_cam_cv_from_world[:3, :3] @ np.array([0.0, 1.0, 0.0])
    up_img = up_cam[:2]
    return int(min(range(4), key=lambda k: _rotate_dir(up_img, k)[1]))   # наименьший y = выше всего


def rotate_image(image: Image.Image, k: int) -> Image.Image:
    return image.rotate(90 * (k % 4), expand=True) if k % 4 else image


def points_to_rotated(points, k: int, width: int, height: int):
    """Пиксели исходного кадра (W×H) → пиксели развёрнутого кадра."""
    p = np.asarray(points, dtype=float)
    x, y = p[..., 0], p[..., 1]
    k %= 4
    if k == 1:
        return np.stack([y, width - x], axis=-1)
    if k == 2:
        return np.stack([width - x, height - y], axis=-1)
    if k == 3:
        return np.stack([height - y, x], axis=-1)
    return p.copy()


def points_from_rotated(points, k: int, width: int, height: int):
    """Пиксели развёрнутого кадра → пиксели исходного кадра (W×H)."""
    p = np.asarray(points, dtype=float)
    x, y = p[..., 0], p[..., 1]
    k %= 4
    if k == 1:
        return np.stack([width - y, x], axis=-1)
    if k == 2:
        return np.stack([width - x, height - y], axis=-1)
    if k == 3:
        return np.stack([y, height - x], axis=-1)
    return p.copy()


def camera_vector_from_rotated(v, k: int):
    """Вектор в осях развёрнутой камеры → оси исходной камеры (OpenCV)."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    for _ in range(k % 4):
        x, y = -y, x
    return np.array([x, y, z])
