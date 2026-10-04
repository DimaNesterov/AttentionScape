"""
AttentionScape — направление взгляда в мировой системе W.

Три источника, от точного к грубому:
    face — модель поворота головы по лицу (6DRepNet), конус 15°;
    body — направление корпуса по линии плеч в 3D, конус 30°;
    away — «смотрит от камеры», конус 45°.
"""
import numpy as np

from .io import Frame

KP_CONF = 0.35
UP = np.array([0.0, 1.0, 0.0])

# метод: (σ конуса в радианах, вес замера в карте внимания)
METHODS = {
    "face": (np.radians(15.0), 1.0),
    "body": (np.radians(30.0), 0.6),
    "away": (np.radians(45.0), 0.3),
}


def face_visible(kc, min_conf: float = 0.5) -> bool:
    """Лицо достаточно видно: уверенно найдены нос и хотя бы один глаз."""
    return kc[0] > min_conf and max(kc[1], kc[2]) > min_conf


def face_box(kxy, kc, width: int, height: int):
    """Плотная рамка вокруг лица по расстоянию уши/глаза/плечи. Возвращает ((x0,y0,x1,y1), центр) или None."""
    head = [kxy[j] for j in range(5) if kc[j] > KP_CONF]
    if not head:
        return None
    center = np.mean(head, axis=0)
    sizes = []
    if kc[3] > KP_CONF and kc[4] > KP_CONF:
        sizes.append(np.linalg.norm(kxy[3] - kxy[4]) * 1.25)
    if kc[1] > KP_CONF and kc[2] > KP_CONF:
        sizes.append(np.linalg.norm(kxy[1] - kxy[2]) * 2.6)
    if kc[5] > KP_CONF and kc[6] > KP_CONF:
        sizes.append(np.linalg.norm(kxy[5] - kxy[6]) * 0.45)
    half = max(max(sizes) if sizes else 0.0, 16.0) * 0.75
    x0, y0 = int(max(0, center[0] - half)), int(max(0, center[1] - half * 0.95))
    x1, y1 = int(min(width, center[0] + half)), int(min(height, center[1] + half * 1.15))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    return (x0, y0, x1, y1), center


def _rotation_z_to(v):
    """Поворот, переводящий ось +Z в направление v (формула Родрига)."""
    a = np.array([0.0, 0.0, 1.0])
    b = v / np.linalg.norm(v)
    c = float(a @ b)
    w = np.cross(a, b)
    s = float(np.linalg.norm(w))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + W + W @ W * ((1 - c) / s ** 2)


def local_gaze_from_angles(pitch_deg: float, yaw_deg: float):
    """Углы 6DRepNet → взгляд в «осях вырезки» (камера смотрит на голову вдоль +Z, OpenCV).
    Соглашение — как в draw_axis из 6DRepNet: yaw = pitch = 0 — смотрит прямо в камеру."""
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    return np.array([-np.sin(y), -np.cos(y) * np.sin(p), -np.cos(y) * np.cos(p)])


def gaze_world_from_local(frame: Frame, g_local, crop_center_uv):
    """Взгляд в осях вырезки → единичный вектор в W (с поправкой: голова не в центре кадра)."""
    ray = np.linalg.inv(frame.K) @ np.array([crop_center_uv[0], crop_center_uv[1], 1.0])
    g_cam = _rotation_z_to(ray) @ np.asarray(g_local, dtype=float)
    g_world = frame.T_world_from_cam_cv[:3, :3] @ g_cam
    return g_world / np.linalg.norm(g_world)


def gaze_from_head_pose(frame: Frame, pitch_deg: float, yaw_deg: float, crop_center_uv):
    return gaze_world_from_local(frame, local_gaze_from_angles(pitch_deg, yaw_deg), crop_center_uv)


def _tilt_down(direction, degrees: float):
    d = np.array(direction, dtype=float)
    d[1] = 0.0
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    d /= n
    a = np.radians(degrees)
    return d * np.cos(a) + np.array([0.0, -np.sin(a), 0.0])


def gaze_from_body(frame: Frame, kxy, kc, head_w, min_width: float = 0.15, pitch_down: float = 10.0):
    """Направление корпуса: плечи переносятся в 3D на глубину головы, «вперёд» = вверх × (правое − левое)."""
    if kc[5] <= KP_CONF or kc[6] <= KP_CONF:
        return None
    T_cw = frame.T_cam_cv_from_world
    z = float((T_cw[:3, :3] @ np.asarray(head_w) + T_cw[:3, 3])[2])
    if z <= 0.1:
        return None
    K_inv = np.linalg.inv(frame.K)
    T_wc = frame.T_world_from_cam_cv

    def to_world(uv):
        pc = z * (K_inv @ np.array([uv[0], uv[1], 1.0]))
        return T_wc[:3, :3] @ pc + T_wc[:3, 3]

    shoulder_axis = to_world(kxy[6]) - to_world(kxy[5])        # правое плечо − левое (COCO 6 и 5)
    shoulder_axis[1] = 0.0
    if np.linalg.norm(shoulder_axis) < min_width:              # человек в профиль — ось плеч неоднозначна
        return None
    return _tilt_down(np.cross(UP, shoulder_axis), pitch_down)


def gaze_away(frame: Frame, head_w, pitch_down: float = 10.0):
    """Запасной вариант: человек смотрит от камеры (вид со спины)."""
    camera = frame.T_world_from_cam_ar[:3, 3]
    return _tilt_down(np.asarray(head_w) - camera, pitch_down)
