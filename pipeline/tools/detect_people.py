"""
AttentionScape — неделя 3, шаг 1: люди на кадрах → положение головы в 3D-модели комнаты.

Запуск из корня репозитория:
    python pipeline/tools/detect_people.py data/sessions/<session_id>

Что делает для каждого записанного кадра:
    1) находит людей и ключевые точки тела (YOLO11-pose);
    2) по лучу через щиколотки и плоскости пола находит, где человек стоит;
       по лучу через голову — где голова (высота над полом);
       если ноги не видны — запасной вариант: голова на типичной высоте 1,55 м;
    3) связывает людей между кадрами по положению в 3D (простой трекер);
    4) пишет perception/people.json и размеченные кадры perception/annotated_*.jpg.

Затем: python pipeline/tools/view_people.py data/sessions/<session_id>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # папка pipeline/
from ascape.geometry import (closest_on_ray_to_vertical, floor_height,  # noqa: E402
                             intersect_horizontal, pixel_ray)
from ascape.io import load_frames, load_mesh  # noqa: E402
from ascape.orient import points_from_rotated, points_to_rotated, rotate_image, upright_rotation  # noqa: E402

KP_CONF = 0.35
HEAD_KP = (0, 1, 2, 3, 4)          # нос, глаза, уши (COCO)
ANKLES = (15, 16)
ANKLE_HEIGHT = 0.08                # щиколотка примерно на 8 см выше пола
HEAD_PRIOR = 1.55                  # типичная высота центра головы стоящего человека
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
TRACK_COLORS = [(230, 60, 60), (60, 160, 230), (240, 180, 40), (170, 90, 220), (40, 200, 140), (240, 120, 180)]


def head_pixel(kxy, kc, box):
    pts = [kxy[i] for i in HEAD_KP if kc[i] > KP_CONF]
    if pts:
        return np.mean(pts, axis=0), True
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, y1 + 0.08 * (y2 - y1)]), False


def foot_pixel(kxy, kc, box, image_height):
    ankles = [kxy[i] for i in ANKLES if kc[i] > KP_CONF]
    if ankles:
        return np.mean(ankles, axis=0), ANKLE_HEIGHT
    x1, y1, x2, y2 = box
    if y2 < image_height - 3:          # низ рамки не обрезан краем кадра
        return np.array([(x1 + x2) / 2, y2]), 0.0
    return None, None


def locate_person(frame, kxy_up, kc, box_up, up_height, to_frame, y_floor):
    """3D-положение головы и ступней в W.
    Точки головы/ступней выбираются в развёрнутой («верхом вверх») картинке, затем
    переводятся в пиксели исходного кадра (to_frame) и превращаются в лучи."""
    head_up, head_from_kp = head_pixel(kxy_up, kc, box_up)
    foot_up, foot_h = foot_pixel(kxy_up, kc, box_up, up_height)
    head_uv = to_frame(head_up)
    origin, head_dir = pixel_ray(frame, *head_uv)

    head_w, foot_w, method = None, None, None
    if foot_up is not None:
        o, d = pixel_ray(frame, *to_frame(foot_up))
        foot_w = intersect_horizontal(o, d, y_floor + foot_h)
        if foot_w is not None:
            foot_w = foot_w.copy()
            foot_w[1] = y_floor
            head_w = closest_on_ray_to_vertical(origin, head_dir, foot_w)
            method = "floor" if head_w is not None else None
    if head_w is None:
        head_w = intersect_horizontal(origin, head_dir, y_floor + HEAD_PRIOR)
        method = "height_prior" if head_w is not None else None

    out = {"head_px": [round(float(head_uv[0]), 1), round(float(head_uv[1]), 1)],
           "head_from_keypoints": bool(head_from_kp), "method": method,
           "head_w": None, "foot_w": None, "height_m": None, "distance_m": None, "plausible": False}
    if head_w is not None:
        height = float(head_w[1] - y_floor)
        out.update(head_w=[round(float(x), 3) for x in head_w],
                   foot_w=None if foot_w is None else [round(float(x), 3) for x in foot_w],
                   height_m=round(height, 2),
                   distance_m=round(float(np.linalg.norm(head_w - origin)), 2),
                   plausible=bool(0.7 <= height <= 2.1))
    return out


class Tracker3D:
    """Связывает людей между кадрами по положению головы в плане (подходит для 1 кадр/с)."""

    def __init__(self, gate_m=2.0, max_gap=3):
        self.gate, self.max_gap = gate_m, max_gap
        self.tracks, self.next_id = [], 1

    def update(self, frame_index, people):
        active = [t for t in self.tracks if frame_index - t["last"] <= self.max_gap]
        pairs = []
        for i, p in enumerate(people):
            if p["head_w"] is None:
                continue
            for j, t in enumerate(active):
                dist = float(np.linalg.norm(np.array(p["head_w"])[[0, 2]] - t["pos"][[0, 2]]))
                pairs.append((dist, i, j))
        used_p, used_t = set(), set()
        for dist, i, j in sorted(pairs):
            if dist > self.gate or i in used_p or j in used_t:
                continue
            people[i]["track"] = active[j]["id"]
            active[j]["pos"], active[j]["last"] = np.array(people[i]["head_w"]), frame_index
            used_p.add(i)
            used_t.add(j)
        for i, p in enumerate(people):
            if i in used_p:
                continue
            if p["head_w"] is None:
                p["track"] = None
                continue
            track = {"id": self.next_id, "pos": np.array(p["head_w"]), "last": frame_index}
            self.next_id += 1
            self.tracks.append(track)
            p["track"] = track["id"]


def annotate(image, people, kxy_all, to_vis, font):
    """Рисует людей на (развёрнутой) картинке; to_vis переводит пиксели кадра в пиксели картинки."""
    draw = ImageDraw.Draw(image)
    for p, kxy in zip(people, kxy_all):
        color = TRACK_COLORS[(p["track"] - 1) % len(TRACK_COLORS)] if p["track"] else (180, 180, 180)
        corners = to_vis(np.array([[p["bbox"][0], p["bbox"][1]], [p["bbox"][2], p["bbox"][3]]]))
        x1, y1 = corners.min(axis=0)
        x2, y2 = corners.max(axis=0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        pts = to_vis(np.asarray(kxy))
        for a, b in SKELETON:
            if p["kp_conf"][a] > KP_CONF and p["kp_conf"][b] > KP_CONF:
                draw.line([tuple(pts[a]), tuple(pts[b])], fill=color, width=3)
        hx, hy = to_vis(np.array(p["head_px"]))
        draw.ellipse([hx - 7, hy - 7, hx + 7, hy + 7], outline=(255, 255, 255), width=3)
        label = f"#{p['track']}  " if p["track"] else ""
        if p["height_m"] is not None:
            label += f"h {p['height_m']:.2f} m  d {p['distance_m']:.1f} m  [{p['method']}]"
            if not p["plausible"]:
                label += "  ?"
        else:
            label += "no 3D"
        draw.text((x1, max(0, y1 - 28)), label, fill=color, font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="папка сессии, например data/sessions/20261004-183519-scan")
    ap.add_argument("--model", default="yolo11s-pose.pt", help="модель YOLO pose (скачается сама)")
    ap.add_argument("--conf", type=float, default=0.5, help="порог уверенности для людей")
    ap.add_argument("--width", type=int, default=1280, help="ширина размеченных кадров")
    args = ap.parse_args()

    session = Path(args.session)
    mesh = load_mesh(session)
    frames = load_frames(session)
    if not frames:
        sys.exit("Нет записанных кадров (capture/frames) с tracking_state = normal")
    y_floor = floor_height(mesh)
    print(f"Кадров: {len(frames)}, высота пола в W: {y_floor:+.3f} м")

    out_dir = session / "perception"
    out_dir.mkdir(exist_ok=True)
    model = YOLO(args.model)
    tracker = Tracker3D()
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()

    k0 = upright_rotation(frames[0])
    if k0:
        print(f"Кадры повёрнуты на {90 * k0}° относительно вертикали — разворачиваю перед распознаванием")

    result_frames = []
    for frame in frames:
        image = Image.open(frame.image_path).convert("RGB")
        k = upright_rotation(frame)
        upright = rotate_image(image, k)
        res = model.predict(upright, conf=args.conf, classes=[0], verbose=False)[0]
        people, kxy_list = [], []

        def to_frame(points, k=k, frame=frame):
            return points_from_rotated(np.asarray(points, dtype=float), k, frame.width, frame.height)

        if res.boxes is not None and len(res.boxes) and res.keypoints is not None:
            boxes_up = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            kxy_up_all = res.keypoints.xy.cpu().numpy()
            kc_all = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
                      else np.ones(kxy_up_all.shape[:2]))
            for box_up, score, kxy_up, kc in zip(boxes_up, scores, kxy_up_all, kc_all):
                p = locate_person(frame, kxy_up, kc, box_up, upright.size[1], to_frame, y_floor)
                corners = to_frame(box_up.reshape(2, 2))
                box = np.concatenate([corners.min(axis=0), corners.max(axis=0)])
                kxy = to_frame(kxy_up)
                p.update(bbox=[round(float(c), 1) for c in box], score=round(float(score), 3),
                         kp_conf=[round(float(c), 2) for c in kc],
                         kp_xy=[[round(float(x), 1), round(float(y), 1)] for x, y in kxy])
                people.append(p)
                kxy_list.append(kxy)
        tracker.update(frame.index, people)

        scale = args.width / max(frame.width, frame.height)
        vw, vh = frame.width * scale, frame.height * scale
        vis = rotate_image(image.resize((int(round(vw)), int(round(vh)))), k)

        def to_vis(points, k=k, vw=vw, vh=vh, scale=scale):
            return points_to_rotated(np.asarray(points, dtype=float) * scale, k, vw, vh)

        annotate(vis, people, kxy_list, to_vis, font)
        vis.save(out_dir / f"annotated_{frame.index:06d}.jpg", quality=88)

        result_frames.append({"index": frame.index, "timestamp": frame.timestamp,
                              "camera_w": [round(float(x), 3) for x in frame.T_world_from_cam_ar[:3, 3]],
                              "people": people})
        summary = ", ".join(
            f"#{p['track']} h={p['height_m']} d={p['distance_m']} {p['method']}" if p["height_m"] is not None
            else "без 3D" for p in people) or "людей нет"
        print(f"кадр {frame.index:4d}: {summary}")

    out = {"schema": "attentionscape/people/0.1", "session_id": session.name,
           "floor_y": round(y_floor, 4), "frames": result_frames}
    (out_dir / "people.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\nТреки:")
    for tid in sorted({p["track"] for f in result_frames for p in f["people"] if p["track"]}):
        pts = [p for f in result_frames for p in f["people"] if p["track"] == tid]
        heights = [p["height_m"] for p in pts if p["height_m"] is not None]
        floor_share = np.mean([p["method"] == "floor" for p in pts])
        print(f"  #{tid}: {len(pts)} кадров, средняя высота головы {np.mean(heights):.2f} м, "
              f"по полу {floor_share:.0%}, правдоподобных {np.mean([p['plausible'] for p in pts]):.0%}")
    print(f"\nРезультаты: {out_dir}")


if __name__ == "__main__":
    main()
