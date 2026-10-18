"""
AttentionScape — сглаживание направления взгляда по треку.

При 1 кадре в секунду соседние замеры одного человека отстоят на секунду, и за это
время взгляд может действительно смениться. Поэтому сглаживание осторожное:
    • замер усредняется только с соседями по времени (±1 кадр) и только если они
      отличаются меньше чем на SWITCH_DEG — больший разворот считаем настоящей сменой взгляда;
    • вклад соседа пропорционален надёжности его метода (1 / σ²) и ослаблен вдвое;
    • если оба соседа согласны между собой (< AGREE_DEG), а центральный замер расходится
      с каждым больше чем на SWITCH_DEG — он «подозрительный»: направление не трогаем
      (это мог быть настоящий короткий взгляд), но снижаем его вес в карте.
"""
import numpy as np

SWITCH_DEG = 35.0
AGREE_DEG = 20.0
MAX_GAP_S = 1.6          # при 1 кадре/с — только непосредственные соседи
NEIGHBOR_FACTOR = 0.5
SUSPECT_WEIGHT = 0.3


def _angle_deg(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def smooth_track(observations):
    """observations — замеры одного трека, отсортированные по времени
    (словари с gaze (единичный вектор), sigma (рад), timestamp).
    Возвращает список (новое направление, метка, множитель веса)."""
    n = len(observations)
    reliability = [1.0 / max(np.degrees(o["sigma"]), 1e-3) ** 2 for o in observations]
    results = []
    for i, ob in enumerate(observations):
        g = np.asarray(ob["gaze"], dtype=float)
        neighbors = [j for j in (i - 1, i + 1)
                     if 0 <= j < n and abs(observations[j]["timestamp"] - ob["timestamp"]) <= MAX_GAP_S]

        if len(neighbors) == 2:
            a = np.asarray(observations[neighbors[0]]["gaze"], dtype=float)
            b = np.asarray(observations[neighbors[1]]["gaze"], dtype=float)
            if _angle_deg(a, b) < AGREE_DEG and _angle_deg(g, a) > SWITCH_DEG and _angle_deg(g, b) > SWITCH_DEG:
                results.append((g, "suspect", SUSPECT_WEIGHT))
                continue

        total = reliability[i] * g
        used = 0
        for j in neighbors:
            gj = np.asarray(observations[j]["gaze"], dtype=float)
            if _angle_deg(g, gj) <= SWITCH_DEG:
                total = total + NEIGHBOR_FACTOR * reliability[j] * gj
                used += 1
        results.append((total / np.linalg.norm(total), "smoothed" if used else "raw", 1.0))
    return results


def smooth_observations(observations):
    """Сглаживает все замеры по трекам на месте: gaze → сглаженный, gaze_raw → исходный,
    weight умножается для подозрительных. Замеры без трека не трогаются. Возвращает счётчики меток."""
    counts = {"smoothed": 0, "raw": 0, "suspect": 0, "no_track": 0}
    tracks = {}
    for ob in observations:
        ob["gaze_raw"] = np.asarray(ob["gaze"], dtype=float)
        ob["smoothing"] = "no_track"
        if ob.get("track") is None:
            counts["no_track"] += 1
            continue
        tracks.setdefault(ob["track"], []).append(ob)
    for obs in tracks.values():
        obs.sort(key=lambda o: o["timestamp"])
        for ob, (g, label, factor) in zip(obs, smooth_track(obs)):
            ob["gaze"] = g
            ob["smoothing"] = label
            ob["weight"] = ob["weight"] * factor
            counts[label] += 1
    return counts