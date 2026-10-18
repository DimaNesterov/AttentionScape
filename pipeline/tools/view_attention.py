"""
AttentionScape — просмотр карты внимания в 3D (браузер).

Запуск из корня репозитория (после gaze_attention.py):
    python pipeline/tools/view_attention.py data/sessions/<session_id>

Цвета меша:
    тёмно-серый — сюда не смотрел никто (неизвестно);
    светло-серый — в поле зрения было, но внимания мало;
    синий → голубой → зелёный → жёлтый → оранжевый → красный — от слабого внимания к максимальному.
Линии — центральные лучи взгляда (голубые — по глазам, зелёные — по повороту головы,
жёлтые — по плечам, серые — «от камеры»).
Белый куб с лучом — камера (где стоял телефон и куда смотрел).
Названия комнат (Mark room) — подписи над полом с долей внимания.
Клавиша R — скрыть/показать лучи. Клик по голове (шарику) — номер кадра и ссылка на кадр со стрелкой.
Надписи в браузере — на английском (для портфолио).
"""
import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # папка pipeline/
from ascape.geometry import floor_height  # noqa: E402
from ascape.io import load_frames, load_mesh  # noqa: E402

from ascape.rooms import load_rooms  # noqa: E402
from view_mesh import build_viewer_html  # noqa: E402

# Многоцветная шкала: синий → голубой → зелёный → жёлтый → оранжевый → красный.
CMAP_STOPS = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.88, 1.0])
CMAP = np.array([[35, 23, 125], [40, 90, 220], [30, 170, 230], [40, 210, 140],
                 [150, 225, 50], [250, 200, 40], [245, 110, 30], [210, 25, 25]], dtype=float)
MIN_LEVEL = 0.06   # слабее этого — не красим (хвосты конусов), остаётся серый
METHOD_HEX = {"eyes": "0x3cc8ff", "head": "0x3cdc5a", "body": "0xf0be28", "away": "0xaaaaaa"}

EXTRA_JS = """
const CAMS = __CAMS__;
for (const c of CAMS) {
  const box = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 0.06), new THREE.MeshBasicMaterial({color: 0xffffff}));
  box.position.set(c.p[0], c.p[1], c.p[2]);
  scene.add(box);
  const end = new THREE.Vector3(c.p[0] + c.f[0] * 0.6, c.p[1] + c.f[1] * 0.6, c.p[2] + c.f[2] * 0.6);
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([box.position.clone(), end]),
                           new THREE.LineBasicMaterial({color: 0xffffff})));
}
const ROOMS = __ROOMS__;
for (const r of ROOMS) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 56px -apple-system, Helvetica, Arial, sans-serif';
  canvas.width = Math.ceil(ctx.measureText(r.label).width) + 40;
  canvas.height = 84;
  ctx.font = 'bold 56px -apple-system, Helvetica, Arial, sans-serif';
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#ffffff';
  ctx.textBaseline = 'middle';
  ctx.fillText(r.label, 20, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: tex, depthTest: false}));
  sprite.scale.set(0.35 * canvas.width / canvas.height, 0.35, 1);
  sprite.position.set(r.p[0], r.p[1], r.p[2]);
  sprite.renderOrder = 10;
  scene.add(sprite);
}
const OBS = __OBS__;
const rays = new THREE.Group();
scene.add(rays);
window.addEventListener('keydown', e => { if (e.key === 'r' || e.key === 'R') rays.visible = !rays.visible; });
const headGeo = new THREE.SphereGeometry(0.06, 12, 8);
const heads = [];
for (const o of OBS) {
  const color = Number(o.color);
  const eye = new THREE.Vector3(o.eye[0], o.eye[1], o.eye[2]);
  const end = o.hit ? new THREE.Vector3(o.hit[0], o.hit[1], o.hit[2])
                    : eye.clone().add(new THREE.Vector3(o.gaze[0], o.gaze[1], o.gaze[2]).multiplyScalar(1.5));
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([eye, end]),
                              new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.8}));
  rays.add(line);
  const s = new THREE.Mesh(headGeo, new THREE.MeshBasicMaterial({color}));
  s.position.copy(eye);
  s.userData = {o, line};
  rays.add(s);
  heads.push(s);
}

// Click a head (sphere) to see which frame it came from; the link opens that frame with the gaze arrow.
const pick = document.createElement('div');
pick.style.cssText = 'position:absolute;bottom:12px;left:12px;background:rgba(0,0,0,.7);padding:8px 12px;border-radius:8px;display:none';
document.body.appendChild(pick);
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let down = null, selected = null;
renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', e => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5 || !rays.visible) return;
  pointer.set(e.clientX / window.innerWidth * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(heads)[0];
  if (selected) { selected.line.material.linewidth = 1; selected.line.material.color.set(Number(selected.o.color)); selected = null; }
  if (!hit) { pick.style.display = 'none'; return; }
  selected = hit.object.userData;
  selected.line.material.color.set(0xffffff);
  const o = selected.o;
  const name = String(o.frame).padStart(6, '0');
  pick.innerHTML = `<b>Frame ${o.frame}</b> · take ${o.take} · person #${o.track ?? '–'} · ${o.label}` +
    (o.angles ? ` · face ${o.angles.face_px}px` : '') + `<br><a style="color:#7cc4ff" target="_blank" href="../perception/gaze_${name}.jpg">open frame with gaze arrow</a>`;
  pick.style.display = 'block';
});
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
    takes = {f.index: f.take for f in load_frames(session)}
    method_label = {"eyes": "eyes", "head": "head pose", "body": "shoulders", "away": "away from camera"}
    obs = [{"eye": o["eye"], "gaze": o["gaze"], "hit": o["hit"], "color": METHOD_HEX.get(o["method"], "0xaaaaaa"),
            "frame": o["frame"], "take": takes.get(o["frame"]), "track": o["track"],
            "label": method_label.get(o["method"], o["method"]), "angles": o.get("angles")}
           for o in gaze]

    # Положения камеры: уникальные (на штативе камера почти неподвижна).
    cams = []
    for fr in load_frames(session)[::10]:
        T = fr.T_world_from_cam_ar
        c = {"p": T[:3, 3].round(3).tolist(), "f": (-T[:3, 2]).round(3).tolist()}   # камера ARKit смотрит вдоль −Z
        if all(np.linalg.norm(np.array(c["p"]) - np.array(u["p"])) > 0.15 for u in cams):
            cams.append(c)
    counts = {m: sum(o["method"] == m for o in gaze) for m in METHOD_HEX}

    # Подписи комнат: на высоте 0,3 м над полом в месте отметки, с долей внимания (если посчитана).
    shares = {r["name"]: r["share"] for r in meta.get("rooms", [])}
    y_label = floor_height(mesh) + 0.3
    rooms = [{"label": r["name"] + (f" {shares[r['name']]:.0%}" if r["name"] in shares else ""),
              "p": [r["position"][0], y_label, r["position"][2]]} for r in load_rooms(session)]
    rooms_info = ""
    if shares:
        rooms_info = "<b>Attention by room</b><br>" + "".join(
            f"{name}: {share:.0%}<br>" for name, share in sorted(shares.items(), key=lambda x: -x[1]))
        if meta.get("rooms_other_share", 0) >= 0.01:
            rooms_info += f"Outside marked rooms: {meta['rooms_other_share']:.0%}<br>"
        rooms_info += "<br>"

    legend = ('<span class="sw" style="background:rgb(70,70,70)"></span>Not observed (unknown)<br>'
              '<span class="sw" style="background:rgb(185,185,185)"></span>In view, little attention<br>'
              '<span class="sw" style="background:linear-gradient(90deg,rgb(35,23,125),rgb(40,90,220),'
              'rgb(30,170,230),rgb(40,210,140),rgb(150,225,50),rgb(250,200,40),rgb(245,110,30),'
              'rgb(210,25,25));width:70px"></span>Attention: low → high<br>')
    extra_info = (rooms_info + "<b>Gaze rays</b><br>"
                  f'<span class="sw" style="background:#3cc8ff"></span>From eyes ({counts["eyes"]})<br>'
                  f'<span class="sw" style="background:#3cdc5a"></span>From head pose ({counts["head"]})<br>'
                  f'<span class="sw" style="background:#f0be28"></span>From shoulders ({counts["body"]})<br>'
                  f'<span class="sw" style="background:#aaaaaa"></span>Away from camera ({counts["away"]})<br>'
                  "White box — camera<br>"
                  "R — show/hide rays · click a head for its frame<br><br>")
    extra_js = (EXTRA_JS.replace("__OBS__", json.dumps(obs)).replace("__CAMS__", json.dumps(cams))
                .replace("__ROOMS__", json.dumps(rooms, ensure_ascii=False)))
    html = build_viewer_html(session, extra_js=extra_js,
                             extra_info=extra_info, verbose=False, vertex_colors=colors, legend_html=legend)
    out = session / "results" / "attention_view.html"
    out.write_text(html, encoding="utf-8")
    print(f"Просмотр: {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()