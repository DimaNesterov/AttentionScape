"""
AttentionScape — 3D-траектории голов людей в модели комнаты.

Запуск из корня репозитория (после detect_people.py):
    python pipeline/tools/view_people.py data/sessions/<session_id>

Показывает меш комнаты, траектории голов по трекам (цветные линии и шары:
яркие — надёжные точки по полу, бледные — по типичной высоте) и камеру
(белый куб с лучом направления).
"""
import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # папка pipeline/
from ascape.io import load_frames  # noqa: E402
from view_mesh import build_viewer_html  # noqa: E402  (тот же каталог tools/)

TRACK_COLORS = [(230, 60, 60), (60, 160, 230), (240, 180, 40), (170, 90, 220), (40, 200, 140), (240, 120, 180)]

EXTRA_JS = """
const TRACKS = __TRACKS__;
const CAMS = __CAMS__;
for (const tr of TRACKS) {
  const col = new THREE.Color(tr.color[0]/255, tr.color[1]/255, tr.color[2]/255);
  const pts = tr.points.map(p => new THREE.Vector3(p.p[0], p.p[1], p.p[2]));
  if (pts.length > 1) {
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
                             new THREE.LineBasicMaterial({color: col})));
  }
  const geo = new THREE.SphereGeometry(0.06, 14, 10);
  const solid = new THREE.MeshBasicMaterial({color: col});
  const faint = new THREE.MeshBasicMaterial({color: col, transparent: true, opacity: 0.35});
  tr.points.forEach((p, i) => {
    const s = new THREE.Mesh(geo, p.floor ? solid : faint);
    s.position.copy(pts[i]);
    scene.add(s);
  });
}
for (const c of CAMS) {
  const box = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 0.06), new THREE.MeshBasicMaterial({color: 0xffffff}));
  box.position.set(c.p[0], c.p[1], c.p[2]);
  scene.add(box);
  const end = new THREE.Vector3(c.p[0] + c.f[0]*0.6, c.p[1] + c.f[1]*0.6, c.p[2] + c.f[2]*0.6);
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([box.position.clone(), end]),
                           new THREE.LineBasicMaterial({color: 0xffffff})));
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="папка сессии")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    session = Path(args.session)
    people_path = session / "perception" / "people.json"
    if not people_path.exists():
        sys.exit("Нет perception/people.json — сначала запустите detect_people.py")
    data = json.loads(people_path.read_text(encoding="utf-8"))

    tracks = {}
    for f in data["frames"]:
        for p in f["people"]:
            if p.get("track") and p.get("head_w"):
                tracks.setdefault(p["track"], []).append({"p": p["head_w"], "floor": p["method"] == "floor"})
    track_list = [{"id": tid, "color": TRACK_COLORS[(tid - 1) % len(TRACK_COLORS)], "points": pts}
                  for tid, pts in sorted(tracks.items())]

    # Положения камеры: уникальные (камера на штативе почти неподвижна).
    cams = []
    for fr in load_frames(session)[::10]:
        T = fr.T_world_from_cam_ar
        forward = -T[:3, 2]                         # камера ARKit смотрит вдоль −Z
        cams.append({"p": T[:3, 3].round(3).tolist(), "f": forward.round(3).tolist()})
    uniq = []
    for c in cams:
        if all(np.linalg.norm(np.array(c["p"]) - np.array(u["p"])) > 0.15 for u in uniq):
            uniq.append(c)

    legend = "".join(
        f'<span class="sw" style="background:rgb{tuple(t["color"])}"></span>трек #{t["id"]} ({len(t["points"])} точек)<br>'
        for t in track_list)
    extra_info = (f"<b>Люди</b><br>{legend or 'треков нет<br>'}"
                  "яркие шары — по полу, бледные — по типичной высоте<br>белый куб — камера<br><br>")
    extra_js = EXTRA_JS.replace("__TRACKS__", json.dumps(track_list)).replace("__CAMS__", json.dumps(uniq))

    html = build_viewer_html(session, extra_js=extra_js, extra_info=extra_info, verbose=False)
    out = session / "perception" / "people_view.html"
    out.write_text(html, encoding="utf-8")
    print(f"Треков: {len(track_list)}, положений камеры: {len(uniq)}")
    print(f"Просмотр: {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
