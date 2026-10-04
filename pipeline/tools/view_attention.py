"""
AttentionScape — просмотр карты внимания в 3D (браузер).

Запуск из корня репозитория (после gaze_attention.py):
    python pipeline/tools/view_attention.py data/sessions/<session_id>

Цвета меша:
    тёмно-серый — сюда не смотрел никто (неизвестно);
    светло-серый — в поле зрения было, но внимания мало;
    синий → голубой → зелёный → жёлтый → оранжевый → красный — от слабого внимания к максимальному.
Линии — центральные лучи взгляда (зелёные — по лицу, жёлтые — по плечам, серые — «от камеры»).
Клавиша R — скрыть/показать лучи.
"""
import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # папка pipeline/
from ascape.io import load_mesh  # noqa: E402
from view_mesh import build_viewer_html  # noqa: E402

# Многоцветная шкала: синий → голубой → зелёный → жёлтый → оранжевый → красный.
CMAP_STOPS = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.88, 1.0])
CMAP = np.array([[35, 23, 125], [40, 90, 220], [30, 170, 230], [40, 210, 140],
                 [150, 225, 50], [250, 200, 40], [245, 110, 30], [210, 25, 25]], dtype=float)
MIN_LEVEL = 0.06   # слабее этого — не красим (хвосты конусов), остаётся серый
METHOD_HEX = {"face": "0x3cdc5a", "body": "0xf0be28", "away": "0xaaaaaa"}

EXTRA_JS = """
const OBS = __OBS__;
const rays = new THREE.Group();
scene.add(rays);
window.addEventListener('keydown', e => { if (e.key === 'r' || e.key === 'R') rays.visible = !rays.visible; });
const headGeo = new THREE.SphereGeometry(0.05, 12, 8);
for (const o of OBS) {
  const color = Number(o.color);
  const eye = new THREE.Vector3(o.eye[0], o.eye[1], o.eye[2]);
  const end = o.hit ? new THREE.Vector3(o.hit[0], o.hit[1], o.hit[2])
                    : eye.clone().add(new THREE.Vector3(o.gaze[0], o.gaze[1], o.gaze[2]).multiplyScalar(1.5));
  rays.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([eye, end]),
                          new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.8})));
  const s = new THREE.Mesh(headGeo, new THREE.MeshBasicMaterial({color}));
  s.position.copy(eye);
  rays.add(s);
}
"""


def colormap(t):
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, CMAP_STOPS, CMAP[:, c]) for c in range(3)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="папка сессии")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    session = Path(args.session)
    meta_path = session / "results" / "attention.json"
    if not meta_path.exists():
        sys.exit("Нет results/attention.json — сначала запустите gaze_attention.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mesh = load_mesh(session)
    nv = len(mesh.vertices)
    if meta["n_vertices"] != nv:
        sys.exit("Карта внимания посчитана для другого меша")
    data = np.fromfile(session / "results" / "attention_f32.bin", dtype="<f4")
    attention = data[:meta["n_time_bins"] * nv].reshape(meta["n_time_bins"], nv).sum(axis=0)
    coverage = data[meta["n_time_bins"] * nv:]

    # Базовый цвет: серый; темнее там, куда никто не смотрел.
    observed = coverage > np.percentile(coverage[coverage > 0], 5) if (coverage > 0).any() else coverage > 0
    base = np.where(observed[:, None], 185.0, 70.0) * np.ones((nv, 3))
    positive = attention[attention > 0]
    if len(positive):
        level = np.clip(attention / np.percentile(positive, 99.5), 0, 1) ** 0.5
        painted = (level >= MIN_LEVEL)[:, None]
        colors = np.where(painted, colormap((level - MIN_LEVEL) / (1 - MIN_LEVEL)), base)
    else:
        colors = base
    colors = colors.clip(0, 255).astype(np.uint8)

    gaze = json.loads((session / "perception" / "gaze.json").read_text(encoding="utf-8"))["observations"]
    obs = [{"eye": o["eye"], "gaze": o["gaze"], "hit": o["hit"], "color": METHOD_HEX[o["method"]]} for o in gaze]
    counts = {m: sum(o["method"] == m for o in gaze) for m in METHOD_HEX}

    legend = ('<span class="sw" style="background:rgb(70,70,70)"></span>не наблюдалось (неизвестно)<br>'
              '<span class="sw" style="background:rgb(185,185,185)"></span>в поле зрения, внимания мало<br>'
              '<span class="sw" style="background:linear-gradient(90deg,rgb(35,23,125),rgb(40,90,220),'
              'rgb(30,170,230),rgb(40,210,140),rgb(150,225,50),rgb(250,200,40),rgb(245,110,30),'
              'rgb(210,25,25));width:70px"></span>внимание: мало → много<br>')
    extra_info = ("<b>Лучи взгляда</b><br>"
                  f'<span class="sw" style="background:#3cdc5a"></span>по лицу ({counts["face"]})<br>'
                  f'<span class="sw" style="background:#f0be28"></span>по плечам ({counts["body"]})<br>'
                  f'<span class="sw" style="background:#aaaaaa"></span>от камеры ({counts["away"]})<br>'
                  "R — скрыть/показать лучи<br><br>")
    html = build_viewer_html(session, extra_js=EXTRA_JS.replace("__OBS__", json.dumps(obs)),
                             extra_info=extra_info, verbose=False, vertex_colors=colors, legend_html=legend)
    out = session / "results" / "attention_view.html"
    out.write_text(html, encoding="utf-8")
    print(f"Просмотр: {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
