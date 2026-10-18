"""
AttentionScape — разбиение скана на комнаты по отметкам «Mark room» (scan/rooms.json).

Идея: план квартиры как сетка 5 × 5 см. Проходимые клетки — там, где есть пол и нет
препятствий на высоте 0,25–1,8 м (стены, мебель; дверные проёмы остаются проходимыми).
1) «Ядра» комнат: клетки, где до ближайшего препятствия не меньше 0,4 м. Дверные проёмы
   (уже ~0,8 м) в ядра не попадают, поэтому ядра разных комнат не соединены.
2) Отметка комнаты называет ядро, в котором стоит. Если в одном ядре несколько отметок
   (открытая планировка без двери между зонами), ядро делится между ними посередине
   пути пешком — поэтому в такой планировке отметку ставят в центре каждой зоны.
3) Ядра «отращиваются» обратно на всю проходимую площадь; шаг через узкое место дороже,
   поэтому граница комнат проходит по дверям. Ядро без отметки (например, коридор)
   делится между соседними комнатами.
Для комнат с дверями результат не зависит от того, где именно в комнате стоял телефон.
Поверхность (треугольник скана) относится к комнате той клетки, что в 12 см перед ней
со стороны помещения — поэтому две стороны одной стены попадают в разные комнаты.
"""
import heapq
import json
from pathlib import Path

import numpy as np

from .geometry import floor_height
from .io import Mesh

CELL = 0.05                  # м
OBSTACLE_BAND = (0.25, 1.8)  # высота препятствий над полом, м
PROBE = 0.12                 # насколько отступить от поверхности в сторону помещения, м
SNAP = 0.6                   # поверхности дальше этого от размеченных клеток — «вне комнат», м
NARROW = 0.6                 # просвет (расстояние до препятствия), ниже которого шаг дорожает, м
CORE = 0.4                   # минимальный просвет клеток «ядра» комнаты, м
NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]


def load_rooms(session_dir: Path) -> list[dict]:
    path = Path(session_dir) / "scan" / "rooms.json"
    if not path.exists():
        return []
    rooms = json.loads(path.read_text(encoding="utf-8")).get("rooms", [])
    return [r for r in rooms if r.get("name") and len(r.get("position", [])) == 3]


def oriented_face_normals(mesh: Mesh) -> np.ndarray:
    """Единичные нормали треугольников, направленные в помещение (пол — вверх)."""
    v = mesh.vertices.astype(np.float64)
    a, b, c = v[mesh.faces[:, 0]], v[mesh.faces[:, 1]], v[mesh.faces[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    floor, ceiling = mesh.classes == 2, mesh.classes == 3
    evidence = n[floor, 1].sum() if floor.any() else -n[ceiling, 1].sum()
    return -n if evidence < 0 else n


class _Grid:
    def __init__(self, xz_min, xz_max):
        self.origin = np.asarray(xz_min) - 4 * CELL
        size = np.ceil((np.asarray(xz_max) + 4 * CELL - self.origin) / CELL).astype(int) + 1
        self.nx, self.nz = int(size[0]), int(size[1])

    def cell(self, xz):
        ij = np.floor((np.asarray(xz) - self.origin) / CELL).astype(int)
        ij[..., 0] = np.clip(ij[..., 0], 0, self.nx - 1)
        ij[..., 1] = np.clip(ij[..., 1], 0, self.nz - 1)
        return ij

    def mask_from(self, xz):
        m = np.zeros((self.nx, self.nz), dtype=bool)
        if len(xz):
            ij = self.cell(xz)
            m[ij[:, 0], ij[:, 1]] = True
        return m


def _dilate(mask, steps=1):
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _propagate(labels, passable, max_cost=np.inf, step_cost=None):
    """Многоисточниковая Дейкстра по сетке. Источники — клетки с labels ≥ 0.
    step_cost — цена входа в клетку (по умолчанию 1). Возвращает (метки, стоимость пути)."""
    nx, nz = labels.shape
    cost = np.full(labels.shape, np.inf)
    out = labels.copy()
    heap = []
    for i, j in zip(*np.nonzero(labels >= 0)):
        cost[i, j] = 0.0
        heap.append((0.0, int(i), int(j)))
    heapq.heapify(heap)
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > cost[i, j]:
            continue
        for di, dj, w in NEIGHBORS:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < nz and passable[a, b]:
                nd = d + w * (1.0 if step_cost is None else step_cost[a, b])
                if nd < cost[a, b] and nd <= max_cost:
                    cost[a, b] = nd
                    out[a, b] = out[i, j]
                    heapq.heappush(heap, (nd, a, b))
    return out, cost


def _components(mask):
    """Связные области маски (4-связность): номер области или −1."""
    out = np.full(mask.shape, -1, dtype=np.int64)
    nx, nz = mask.shape
    current = 0
    for i0, j0 in zip(*np.nonzero(mask)):
        if out[i0, j0] >= 0:
            continue
        stack = [(int(i0), int(j0))]
        out[i0, j0] = current
        while stack:
            i, j = stack.pop()
            for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= a < nx and 0 <= b < nz and mask[a, b] and out[a, b] < 0:
                    out[a, b] = current
                    stack.append((a, b))
        current += 1
    return out


def segment_rooms(mesh: Mesh, rooms: list[dict]):
    """Возвращает (комната каждого треугольника: индекс в rooms или −1, сетка-план комнат)."""
    floor_y = floor_height(mesh)
    verts = mesh.vertices.astype(np.float64)
    tri = verts[mesh.faces]                              # (M, 3, 3)
    centroids = tri.mean(axis=1)
    normals = oriented_face_normals(mesh)

    grid = _Grid(verts[:, [0, 2]].min(axis=0), verts[:, [0, 2]].max(axis=0))

    # Пол: почти горизонтальные треугольники у самого пола (по вершинам и центрам).
    low = (centroids[:, 1] < floor_y + 0.15) & (np.abs(normals[:, 1]) > 0.7)
    floor_pts = np.concatenate([tri[low].reshape(-1, 3), centroids[low]])[:, [0, 2]]
    floor_mask = _dilate(grid.mask_from(floor_pts), 1)

    # Препятствия: всё, что попадает в полосу высот 0,25–1,8 м над полом.
    lo, hi = floor_y + OBSTACLE_BAND[0], floor_y + OBSTACLE_BAND[1]
    pts = np.concatenate([tri.reshape(-1, 3), centroids])
    band = pts[(pts[:, 1] > lo) & (pts[:, 1] < hi)][:, [0, 2]]
    obstacle = _dilate(grid.mask_from(band), 1)          # закрывает мелкие дыры скана в стенах

    passable = floor_mask & ~obstacle

    # Просвет каждой клетки: расстояние (м) до ближайшей непроходимой клетки.
    blocked = np.where(~passable, 0, -1)
    _, clearance = _propagate(blocked, np.ones_like(passable))
    clearance = np.minimum(clearance, 100.0) * CELL
    narrow = np.clip((NARROW - clearance) / NARROW, 0, 1)
    step_cost = 1.0 + 8.0 * narrow ** 2                                   # дверь ~ в несколько раз дороже

    core = passable & (clearance >= CORE)
    components = _components(core)

    # Отметка → ядро, в котором она стоит (или ближайшее в пределах 1,5 м); иначе — сама клетка.
    seeds = np.full(core.shape, -1, dtype=np.int64)
    core_ij = np.argwhere(core)
    pass_ij = np.argwhere(passable)
    for k, room in enumerate(rooms):
        ij = grid.cell(np.array([room["position"][0], room["position"][2]]))
        if core[ij[0], ij[1]]:
            seeds[ij[0], ij[1]] = k
        elif len(core_ij) and np.linalg.norm(core_ij - ij, axis=1).min() * CELL <= 1.5:
            seeds[tuple(core_ij[np.argmin(np.linalg.norm(core_ij - ij, axis=1))])] = k
        elif len(pass_ij) and np.linalg.norm(pass_ij - ij, axis=1).min() * CELL <= 1.0:
            seeds[tuple(pass_ij[np.argmin(np.linalg.norm(pass_ij - ij, axis=1))])] = k

    # Ядро с отметками целиком (или поделённое между отметками) получает их названия.
    labels = np.full(core.shape, -1, dtype=np.int64)
    seeded_components = np.unique(components[(seeds >= 0) & (components >= 0)])
    for c in seeded_components:
        inside = components == c
        part, _ = _propagate(np.where(inside, seeds, -1), inside, step_cost=step_cost)
        labels[inside] = part[inside]
    outside_core = (seeds >= 0) & (components < 0)
    labels[outside_core] = seeds[outside_core]

    plan, _ = _propagate(labels, passable, step_cost=step_cost)           # ядра → вся проходимая площадь
    near, _ = _propagate(plan, np.ones_like(passable), SNAP / CELL)       # + полоса до 0,6 м (стены, мебель)

    probe = centroids + normals * PROBE
    ij = grid.cell(probe[:, [0, 2]])
    face_room = near[ij[:, 0], ij[:, 1]]
    return face_room, plan


def room_shares(face_room, heat, rooms):
    """Доли внимания по комнатам (по сумме внимания треугольников) и доля «вне комнат»."""
    total = float(heat.sum())
    if total <= 0:
        return [{"name": r["name"], "share": 0.0, "position": r["position"]} for r in rooms], 0.0
    shares = [{"name": r["name"],
               "share": round(float(heat[face_room == k].sum() / total), 4),
               "position": [round(float(x), 3) for x in r["position"]]}
              for k, r in enumerate(rooms)]
    other = round(float(heat[face_room < 0].sum() / total), 4)
    return shares, other