"""
AttentionScape — направление взгляда и черновая карта внимания.

Запуск из корня репозитория (после detect_people.py):
    python pipeline/tools/gaze_attention.py data/sessions/<session_id>

Шаги:
    1) для каждого найденного человека — направление взгляда в W:
       если лицо видно — смесь «по глазам» (MobileGaze) и «по повороту головы» (6DRepNet),
       вес глаз растёт с размером лица; иначе body (линия плеч) → away (от камеры);
    2) кадры со стрелками взгляда (исходные, до сглаживания): perception/gaze_*.jpg;
       затем сглаживание направлений по треку (ascape/smoothing.py; отключается --no-smooth);
    3) карта внимания на меше с учётом перекрытий: results/attention.json + attention_f32.bin;
    4) perception/gaze.json — все замеры (для просмотрщика).

Затем: python pipeline/tools/view_attention.py data/sessions/<session_id>
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # папка pipeline/
from ascape.attention import accumulate, face_areas, faces_to_vertices, write_results  # noqa: E402
from ascape.gaze import (METHODS, blend_directions, blended_sigma, eye_weight,  # noqa: E402
                         face_box, face_visible, gaze_away, gaze_from_body, gaze_world_from_local,
                         local_gaze_from_angles, local_gaze_from_eye_model)
from ascape.orient import (camera_vector_from_rotated, points_from_rotated,  # noqa: E402
                           points_to_rotated, rotate_image, upright_rotation)
from ascape.io import load_frames, load_mesh, load_session, project_raw  # noqa: E402
from ascape.rooms import load_rooms, room_shares, segment_rooms  # noqa: E402
from ascape.smoothing import AGREE_DEG, SUSPECT_WEIGHT, SWITCH_DEG, smooth_observations  # noqa: E402

METHOD_COLORS = {"eyes": (60, 200, 255), "head": (60, 220, 90), "body": (240, 190, 40), "away": (170, 170, 170)}
CLASS_NAMES = {0: "none", 1: "wall", 2: "floor", 3: "ceiling", 4: "table", 5: "seat", 6: "window", 7: "door"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="папка сессии")
    ap.add_argument("--width", type=int, default=1280, help="ширина кадров со стрелками")
    ap.add_argument("--no-smooth", action="store_true", help="не сглаживать направления по треку")
    args = ap.parse_args()

    session = Path(args.session)
    people_path = session / "perception" / "people.json"
    if not people_path.exists():
        sys.exit("Нет perception/people.json — сначала запустите detect_people.py")
    people = json.loads(people_path.read_text(encoding="utf-8"))
    if not any("kp_xy" in p for f in people["frames"] for p in f["people"]):
        sys.exit("В people.json нет координат ключевых точек — перезапустите обновлённый detect_people.py")

    try:
        from sixdrepnet import SixDRepNet
    except ImportError:
        sys.exit("Не установлен пакет sixdrepnet: uv pip install sixdrepnet")
    head_model = SixDRepNet(gpu_id=-1)          # веса (~150 МБ) скачаются при первом запуске
    try:
        from uniface import MobileGaze
    except ImportError:
        sys.exit('Не установлен пакет uniface: uv pip install "uniface[cpu]"')
    eye_model = MobileGaze()                    # ResNet-34, Gaze360; веса (~80 МБ) скачаются при первом запуске

    mesh = load_mesh(session)
    frames = {f.index: f for f in load_frames(session)}
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()

    observations = []
    out_dir = session / "perception"
    for fr in people["frames"]:
        frame = frames.get(fr["index"])
        if frame is None or not fr["people"]:
            continue
        image = Image.open(frame.image_path).convert("RGB")
        k = upright_rotation(frame)
        upright = rotate_image(image, k)                     # лицо для модели — вертикально
        bgr = np.asarray(upright)[:, :, ::-1]
        up_w, up_h = upright.size
        scale = args.width / max(frame.width, frame.height)
        vw, vh = frame.width * scale, frame.height * scale
        vis = rotate_image(image.resize((int(round(vw)), int(round(vh)))), k)
        draw = ImageDraw.Draw(vis)

        for p in fr["people"]:
            if not p.get("head_w") or not p.get("plausible"):
                continue
            kxy, kc = np.array(p["kp_xy"]), np.array(p["kp_conf"])
            head_w = np.array(p["head_w"])
            gaze, method, angles = None, None, None

            kxy_up = points_to_rotated(kxy, k, frame.width, frame.height)
            box = face_box(kxy_up, kc, up_w, up_h) if face_visible(kc) else None
            sigma = None
            if box is not None:
                (x0, y0, x1, y1), _ = box
                crop = np.ascontiguousarray(bgr[y0:y1, x0:x1])
                pitch, yaw, _roll = head_model.predict(crop)
                pitch, yaw = float(np.squeeze(pitch)), float(np.squeeze(yaw))
                eye = eye_model.estimate(crop)
                w_eye = eye_weight(x1 - x0)
                local = blend_directions(local_gaze_from_angles(pitch, yaw),
                                         local_gaze_from_eye_model(eye.pitch, eye.yaw), w_eye)
                g_local = camera_vector_from_rotated(local, k)
                center = points_from_rotated(np.array([(x0 + x1) / 2, (y0 + y1) / 2]), k, frame.width, frame.height)
                gaze = gaze_world_from_local(frame, g_local, center)
                method = "eyes" if w_eye >= 0.5 else "head"
                sigma = blended_sigma(w_eye)
                angles = {"head_pitch": round(pitch, 1), "head_yaw": round(yaw, 1),
                          "eye_pitch": round(float(np.degrees(eye.pitch)), 1),
                          "eye_yaw": round(float(np.degrees(eye.yaw)), 1),
                          "eye_weight": round(w_eye, 2), "face_px": int(x1 - x0)}
            if gaze is None:
                gaze = gaze_from_body(frame, kxy, kc, head_w)
                method = "body" if gaze is not None else None
            if gaze is None:
                gaze = gaze_away(frame, head_w)
                method = "away" if gaze is not None else None
            if gaze is None:
                continue

            default_sigma, weight = METHODS[method]
            sigma = sigma if sigma is not None else default_sigma
            observations.append({"frame": fr["index"], "timestamp": fr["timestamp"], "track": p.get("track"),
                                 "eye": head_w, "gaze": gaze, "method": method,
                                 "sigma": sigma, "weight": weight, "angles": angles})

            # Стрелка взгляда на кадре: проекция точки на 0,6 м впереди головы.
            ends = np.array([head_w, head_w + 0.6 * gaze])
            u, v, z = project_raw(ends, frame)
            if z[0] > 0.05 and z[1] > 0.05:
                color = METHOD_COLORS[method]
                ab = points_to_rotated(np.array([[u[0], v[0]], [u[1], v[1]]]) * scale, k, vw, vh)
                a, b = tuple(ab[0]), tuple(ab[1])
                draw.line([a, b], fill=color, width=5)
                draw.ellipse([b[0] - 6, b[1] - 6, b[0] + 6, b[1] + 6], fill=color)
                draw.text((a[0] + 8, a[1] - 30), f"#{p.get('track')} {method}", fill=color, font=font)
        vis.save(out_dir / f"gaze_{fr['index']:06d}.jpg", quality=88)

    if not observations:
        sys.exit("Нет ни одного замера взгляда")

    smoothing_counts = None
    if not args.no_smooth:
        smoothing_counts = smooth_observations(observations)

    heat, coverage, hits = accumulate(mesh, observations)
    area = np.maximum(face_areas(mesh), 1e-4)
    attention_v = faces_to_vertices(mesh, heat / area)
    coverage_v = faces_to_vertices(mesh, coverage / area)

    info = load_session(session)
    times = [o["timestamp"] for o in observations]
    params = {"methods": {k: {"sigma_deg": round(float(np.degrees(s)), 1), "weight": w} for k, (s, w) in METHODS.items()},
              "head_pose_model": "6DRepNet (sixdrepnet)", "eye_gaze_model": "MobileGaze ResNet-34 (uniface)",
              "eye_weight_px": [40, 120], "render_size": 128, "bias_correction": None,
              "smoothing": None if args.no_smooth else {"switch_deg": SWITCH_DEG, "agree_deg": AGREE_DEG,
                                                        "suspect_weight": SUSPECT_WEIGHT}}
    # Доли внимания по комнатам (если в приложении ставили Mark room).
    rooms = load_rooms(session)
    extra = None
    if rooms:
        face_room, _ = segment_rooms(mesh, rooms)
        shares, other = room_shares(face_room, heat, rooms)
        extra = {"rooms": shares, "rooms_other_share": other}
    results_dir = write_results(session, info, mesh, attention_v, coverage_v, (min(times), max(times)), params, extra)

    gaze_json = [{"frame": o["frame"], "track": o["track"], "method": o["method"],
                  "eye": np.round(o["eye"], 3).tolist(), "gaze": np.round(o["gaze"], 4).tolist(),
                  "gaze_raw": np.round(o.get("gaze_raw", o["gaze"]), 4).tolist(),
                  "smoothing": o.get("smoothing", "off"), "weight": round(float(o["weight"]), 3),
                  "sigma_deg": round(float(np.degrees(o["sigma"])), 1), "angles": o["angles"], "hit": h}
                 for o, h in zip(observations, hits)]
    (out_dir / "gaze.json").write_text(json.dumps({"schema": "attentionscape/gaze/0.1", "observations": gaze_json},
                                                  indent=1, ensure_ascii=False), encoding="utf-8")

    counts = Counter(o["method"] for o in observations)
    print(f"Замеров взгляда: {len(observations)}  (" + ", ".join(f"{k}: {v}" for k, v in counts.items()) + ")")
    if smoothing_counts is not None:
        print(f"Сглаживание: сглажено {smoothing_counts['smoothed']}, без соседей {smoothing_counts['raw']}, "
              f"подозрительных (вес {SUSPECT_WEIGHT}) {smoothing_counts['suspect']}, без трека {smoothing_counts['no_track']}")
    by_class = np.bincount(np.minimum(mesh.classes, 7), weights=heat, minlength=8)
    total = by_class.sum()
    if total > 0:
        print("Внимание по типам поверхностей: " + ", ".join(
            f"{CLASS_NAMES[c]} {by_class[c] / total:.0%}" for c in np.argsort(-by_class) if by_class[c] / total >= 0.02))
    if extra:
        print("Внимание по комнатам: " + ", ".join(
            f"{r['name']} {r['share']:.0%}" for r in sorted(extra["rooms"], key=lambda r: -r["share"]))
            + (f", вне комнат {extra['rooms_other_share']:.0%}" if extra["rooms_other_share"] >= 0.01 else ""))
    takes = sorted({f.take for f in frames.values()})
    if len(takes) > 1:
        print(f"Записей (мест штатива): {len(takes)}")
    print(f"\nКадры со стрелками: {out_dir}/gaze_*.jpg")
    print(f"Карта внимания:     {results_dir}")


if __name__ == "__main__":
    main()