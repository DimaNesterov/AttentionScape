"""
AttentionScape — чтение данных сессии по docs/data_spec.md.

Используется всеми скриптами в pipeline/tools.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Переход ARKit-камера → OpenCV-камера (data_spec.md, раздел 2).
AR_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class Mesh:
    vertices: np.ndarray        # (N, 3) float32, система W, метры
    faces: np.ndarray           # (M, 3) uint32
    classes: np.ndarray         # (M,) uint8, класс ARKit


@dataclass
class Frame:
    index: int
    timestamp: float
    image_path: Path
    width: int
    height: int
    K: np.ndarray               # (3, 3)
    T_world_from_cam_ar: np.ndarray   # (4, 4), соглашение ARKit
    tracking_state: str

    @property
    def T_world_from_cam_cv(self) -> np.ndarray:
        return self.T_world_from_cam_ar @ AR_TO_CV

    @property
    def T_cam_cv_from_world(self) -> np.ndarray:
        return np.linalg.inv(self.T_world_from_cam_cv)


def read_ply(path: Path) -> Mesh:
    data = Path(path).read_bytes()
    end = data.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"{path}: нет end_header")
    nv = nf = 0
    for line in data[:end].decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            if parts[1] == "vertex":
                nv = int(parts[2])
            elif parts[1] == "face":
                nf = int(parts[2])
    offset = end + len(b"end_header\n")
    vertices = np.frombuffer(data, dtype="<f4", count=nv * 3, offset=offset).reshape(nv, 3)
    offset += nv * 12
    face_dtype = np.dtype([("n", "u1"), ("idx", "<u4", (3,)), ("cls", "u1")])
    faces = np.frombuffer(data, dtype=face_dtype, count=nf, offset=offset)
    if nf and not np.all(faces["n"] == 3):
        raise ValueError(f"{path}: не треугольники — формат не по спецификации")
    return Mesh(vertices.astype(np.float32), faces["idx"].astype(np.uint32), faces["cls"].astype(np.uint8))


def load_session(session_dir: Path) -> dict:
    return json.loads((Path(session_dir) / "session.json").read_text(encoding="utf-8"))


def load_mesh(session_dir: Path) -> Mesh:
    return read_ply(Path(session_dir) / "scan" / "mesh.ply")


def load_frames(session_dir: Path, only_normal: bool = True) -> list[Frame]:
    capture = Path(session_dir) / "capture"
    frames = []
    for js in sorted((capture / "frames").glob("*.json")):
        meta = json.loads(js.read_text(encoding="utf-8"))
        cam = meta["camera"]
        if only_normal and cam.get("tracking_state") != "normal":
            continue
        frames.append(Frame(
            index=meta["index"],
            timestamp=meta["timestamp"],
            image_path=capture / meta["image"]["file"],
            width=meta["image"]["width"],
            height=meta["image"]["height"],
            K=np.array(cam["K"], dtype=np.float64).reshape(3, 3),
            T_world_from_cam_ar=np.array(cam["T_world_from_camera_ar"], dtype=np.float64).reshape(4, 4),
            tracking_state=cam.get("tracking_state", "?"),
        ))
    return frames


def project_raw(points_w: np.ndarray, frame: Frame):
    """Мировые точки → (u, v, z) в пикселях кадра (OpenCV), без отсечения."""
    T = frame.T_cam_cv_from_world
    pc = points_w @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    zs = np.where(np.abs(z) > 1e-6, z, 1e-6)
    u = frame.K[0, 0] * pc[:, 0] / zs + frame.K[0, 2]
    v = frame.K[1, 1] * pc[:, 1] / zs + frame.K[1, 2]
    return u, v, z


def project(points_w: np.ndarray, frame: Frame, min_depth: float = 0.05):
    """Мировые точки → пиксели кадра (OpenCV). Возвращает u, v, глубину и маску видимости."""
    T = frame.T_cam_cv_from_world
    pc = points_w @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    ok = z > min_depth
    zs = np.where(ok, z, 1.0)
    u = frame.K[0, 0] * pc[:, 0] / zs + frame.K[0, 2]
    v = frame.K[1, 1] * pc[:, 1] / zs + frame.K[1, 2]
    ok &= (u >= 0) & (u < frame.width) & (v >= 0) & (v < frame.height)
    return u, v, z, ok
