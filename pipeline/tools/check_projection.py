"""
AttentionScape — главная проверка систем координат (неделя 2).

Накладывает отсканированный меш (цвета по классам) на кадры, записанные
приложением в режиме Record. Если цветные области ложатся на реальные
стены, пол, стол и мебель на фото — позы камеры, внутренние параметры
и переход ARKit → OpenCV верны.

Запуск из корня репозитория:
    python pipeline/tools/check_projection.py data/sessions/<session_id>
    python pipeline/tools/check_projection.py data/sessions/<session_id> --every 3 --width 1280

Результат: data/sessions/<id>/check/proj_<кадр>.jpg (+ открывается первый).
"""
import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # папка pipeline/
from ascape.io import load_frames, load_mesh, project_raw  # noqa: E402

CLASS_RGB = np.array([
    (150, 150, 150), (205, 200, 190), (170, 120, 80), (240, 240, 240),
    (60, 130, 220), (60, 180, 90), (120, 220, 230), (220, 150, 60),
], dtype=np.uint8)


def render_overlay(mesh, frame, out_w: int, alpha: float = 0.45):
    """Рисует треугольники меша поверх кадра: от дальних к ближним, цвет — класс поверхности."""
    out_h = int(round(frame.height * out_w / frame.width))
    scale = out_w / frame.width
    u, v, z = project_raw(mesh.vertices.astype(np.float64), frame)
    zf = z[mesh.faces]
    uf = u[mesh.faces] * scale
    vf = v[mesh.faces] * scale
    visible = ((zf > 0.05).all(axis=1)
               & (uf.max(axis=1) >= 0) & (uf.min(axis=1) < out_w)
               & (vf.max(axis=1) >= 0) & (vf.min(axis=1) < out_h))
    sel = np.nonzero(visible)[0]
    order = sel[np.argsort(-zf[sel].mean(axis=1))]          # дальние сначала, ближние поверх

    color_img = Image.new("RGB", (out_w, out_h))
    mask_img = Image.new("L", (out_w, out_h), 0)
    draw_c = ImageDraw.Draw(color_img)
    draw_m = ImageDraw.Draw(mask_img)
    palette = [tuple(int(x) for x in rgb) for rgb in CLASS_RGB]
    tri_u = uf[order].tolist()
    tri_v = vf[order].tolist()
    tri_c = np.minimum(mesh.classes[order], 7).tolist()
    for us, vs, c in zip(tri_u, tri_v, tri_c):
        pts = list(zip(us, vs))
        draw_c.polygon(pts, fill=palette[c])
        draw_m.polygon(pts, fill=255)

    photo = np.asarray(Image.open(frame.image_path).convert("RGB").resize((out_w, out_h)), dtype=np.float32)
    color = np.asarray(color_img, dtype=np.float32)
    mask = (np.asarray(mask_img) > 0)[..., None]
    blended = np.where(mask, (1 - alpha) * photo + alpha * color, photo)
    return Image.fromarray(blended.astype(np.uint8)), float(mask.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="папка сессии, например data/sessions/20261004-120000-scan")
    ap.add_argument("--every", type=int, default=5, help="брать каждый N-й кадр")
    ap.add_argument("--width", type=int, default=1280, help="ширина выходных картинок")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    session = Path(args.session)
    if not (session / "capture" / "frames").exists():
        sys.exit("В сессии нет capture/frames — сначала запишите кадры кнопкой Record")

    mesh = load_mesh(session)
    frames = load_frames(session)
    if not frames:
        sys.exit("Нет кадров с tracking_state = normal")
    print(f"Меш: {len(mesh.vertices):,} вершин, {len(mesh.faces):,} треугольников; кадров: {len(frames)}")

    out_dir = session / "check"
    out_dir.mkdir(exist_ok=True)
    first = None
    for frame in frames[::args.every]:
        img, coverage = render_overlay(mesh, frame, args.width)
        out = out_dir / f"proj_{frame.index:06d}.jpg"
        img.save(out, quality=90)
        first = first or out
        cam_pos = frame.T_world_from_cam_ar[:3, 3]
        print(f"кадр {frame.index:6d}: покрытие мешем {coverage:5.1%}, камера в W = "
              f"({cam_pos[0]:+.2f}, {cam_pos[1]:+.2f}, {cam_pos[2]:+.2f}) м -> {out.name}")

    print(f"\nКартинки: {out_dir}")
    if first and not args.no_open:
        webbrowser.open(first.resolve().as_uri())


if __name__ == "__main__":
    main()
