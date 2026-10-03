"""
AttentionScape — spike 0: с какой дистанции мы надёжно видим голову и лицо?

Запуск (из папки pipeline/):
    python spikes/head_distance_spike.py --photos ../data/spike0 --out ../data/spike0_out

Имена фотографий: d<метры>_<ракурс>.<расширение>
    ракурс: front | 45 | profile | back | down
    примеры: d4_front.heic, d6_profile.jpg, d2_down.heic

Что делает:
    1) находит людей и ключевые точки тела (YOLO11-pose),
    2) вырезает область головы в полном разрешении 48 Мп,
    3) ищет лицо в вырезке головы и грубо оценивает поворот (OpenCV YuNet),
    4) пишет CSV и сохраняет размеченные картинки.
Это проверка осуществимости, а не финальная модель взгляда.
"""
import argparse
import csv
import re
import urllib.request
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
from ultralytics import YOLO

pillow_heif.register_heif_opener()

FACE_MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx")
HEAD_KP = {"nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4}  # индексы COCO
KP_CONF = 0.3
EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png"}


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def ensure_face_model(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        print("Скачиваю модель детектора лиц YuNet (OpenCV)...")
        urllib.request.urlretrieve(FACE_MODEL_URL, path)
    return path


def head_box(kxy, kconf, W, H):
    """Квадрат вокруг головы по видимым точкам головы и ширине плеч."""
    pts = np.array([kxy[i] for i in HEAD_KP.values() if kconf[i] > KP_CONF])
    if len(pts) == 0:
        return None
    c = pts.mean(axis=0)
    span = np.ptp(pts, axis=0).max() if len(pts) > 1 else 0.0
    shoulders = (np.linalg.norm(kxy[5] - kxy[6])
                 if kconf[5] > KP_CONF and kconf[6] > KP_CONF else 0.0)
    size = max(span * 1.8, shoulders * 0.7, 24.0)
    x0, y0 = max(0, c[0] - size), max(0, c[1] - size * 1.1)
    x1, y1 = min(W, c[0] + size), min(H, c[1] + size * 0.9)
    return int(x0), int(y0), int(x1), int(y1)


def face_pose(detector, crop: Image.Image):
    """Лицо в вырезке головы (YuNet). Возвращает (score, yaw_proxy, pitch_proxy) или None.
    yaw_proxy ~ 0 при взгляде в камеру, растёт по модулю при повороте головы (грубый показатель)."""
    scale = 320 / max(crop.size)
    if scale > 1:
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.BICUBIC)
    bgr = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    f = faces[np.argmax(faces[:, 14])]
    re_, le_, nose = f[4:6], f[6:8], f[8:10]
    mid = (re_ + le_) / 2
    ied = max(float(np.linalg.norm(le_ - re_)), 1e-6)
    return float(f[14]), float((nose[0] - mid[0]) / ied), float((nose[1] - mid[1]) / ied)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True, help="папка с фотографиями")
    ap.add_argument("--out", required=True, help="куда сохранить CSV и разметку")
    ap.add_argument("--imgsz", type=int, default=1920, help="размер входа детектора людей")
    args = ap.parse_args()

    photos = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in EXTS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pose_model = YOLO("yolo11n-pose.pt")  # веса скачаются сами при первом запуске
    face_path = ensure_face_model(Path(__file__).resolve().parent / "models")
    detector = cv2.FaceDetectorYN.create(str(face_path), "", (320, 320), 0.6, 0.3, 5000)
    try:
        font = ImageFont.load_default(size=48)
    except TypeError:
        font = ImageFont.load_default()

    rows = []
    for p in photos:
        m = re.match(r"d(\d+(?:\.\d+)?)_([a-z0-9]+)", p.stem.lower())
        dist, orient = (float(m.group(1)), m.group(2)) if m else (None, "?")
        img = load_image(p)
        W, H = img.size
        res = pose_model(img, imgsz=args.imgsz, verbose=False)[0]
        vis = img.copy()
        draw = ImageDraw.Draw(vis)

        n = 0 if res.boxes is None else len(res.boxes)
        if n == 0 or res.keypoints is None:
            rows.append(dict(file=p.name, distance_m=dist, orientation=orient, person=-1))
            print(f"{p.name}: человек не найден")
        else:
            kxy_all = res.keypoints.xy.cpu().numpy()
            kc_all = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
                      else np.ones(kxy_all.shape[:2]))
            boxes = res.boxes.xyxy.cpu().numpy()
            for i in range(n):
                kxy, kc = kxy_all[i], kc_all[i]
                row = dict(file=p.name, distance_m=dist, orientation=orient, person=i,
                           person_height_px=round(float(boxes[i][3] - boxes[i][1])),
                           **{f"conf_{k}": round(float(kc[j]), 2) for k, j in HEAD_KP.items()})
                hb = head_box(kxy, kc, W, H)
                if hb is None:
                    row.update(head_px=None, face_found=False)
                else:
                    row["head_px"] = hb[2] - hb[0]
                    pose = face_pose(detector, img.crop(hb))
                    row["face_found"] = pose is not None
                    if pose:
                        row.update(face_score=round(pose[0], 2), yaw_proxy=round(pose[1], 2),
                                   pitch_proxy=round(pose[2], 2))
                    color = (0, 200, 0) if pose else (230, 120, 0)
                    draw.rectangle(hb, outline=color, width=6)
                    label = (f"face {pose[0]:.2f}  yaw~{pose[1]:+.2f}" if pose else "no face")
                    draw.text((hb[0], max(0, hb[1] - 60)), label, fill=color, font=font)
                rows.append(row)
                print(f"{p.name}: человек {i}, голова {row.get('head_px')} px, лицо: {row.get('face_found')}")

        vis.thumbnail((2400, 2400))
        vis.save(out / f"annotated_{p.stem}.jpg", quality=88)

    fields = ["file", "distance_m", "orientation", "person", "person_height_px", "head_px",
              "conf_nose", "conf_l_eye", "conf_r_eye", "conf_l_ear", "conf_r_ear",
              "face_found", "face_score", "yaw_proxy", "pitch_proxy"]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nГотово: {out / 'results.csv'} и размеченные картинки в {out}")


if __name__ == "__main__":
    main()
