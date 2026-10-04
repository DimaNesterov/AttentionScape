"""
AttentionScape — просмотр отсканированного меша в 3D (браузер).

Запуск из корня репозитория:
    python pipeline/tools/view_mesh.py data/sessions/<session_id>

Что делает:
    1) читает scan/mesh.ply в формате docs/data_spec.md,
    2) печатает статистику (вершины, треугольники, классы, размеры комнаты),
    3) пишет scan/mesh_view.html и открывает его в браузере.

Управление в браузере: левая кнопка — вращать, колесо — масштаб,
правая кнопка — сдвиг, C — показать/скрыть потолок, W — каркас.
"""
import argparse
import base64
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

# Классы ARKit (ARMeshClassification) и цвета для отображения.
CLASSES = {
    0: ("none", (150, 150, 150)),
    1: ("wall", (205, 200, 190)),
    2: ("floor", (170, 120, 80)),
    3: ("ceiling", (240, 240, 240)),
    4: ("table", (60, 130, 220)),
    5: ("seat", (60, 180, 90)),
    6: ("window", (120, 220, 230)),
    7: ("door", (220, 150, 60)),
}


def read_ply(path: Path):
    data = path.read_bytes()
    end = data.find(b"end_header\n")
    if end < 0:
        sys.exit("Не найден конец заголовка PLY (end_header)")
    header = data[:end].decode("utf-8")
    nv = nf = 0
    for line in header.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            if parts[1] == "vertex":
                nv = int(parts[2])
            elif parts[1] == "face":
                nf = int(parts[2])
    offset = end + len(b"end_header\n")
    verts = np.frombuffer(data, dtype="<f4", count=nv * 3, offset=offset).reshape(nv, 3)
    offset += nv * 12
    face_dtype = np.dtype([("n", "u1"), ("idx", "<u4", (3,)), ("cls", "u1")])  # 14 байт, без выравнивания
    faces = np.frombuffer(data, dtype=face_dtype, count=nf, offset=offset)
    if nf and not np.all(faces["n"] == 3):
        sys.exit("В PLY встретились не треугольники — формат не совпадает со спецификацией")
    return verts, faces["idx"].astype(np.uint32), faces["cls"].astype(np.uint8)


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>AttentionScape — __TITLE__</title>
<style>
 html,body{margin:0;height:100%;background:#15181d;color:#e8e8e8;font:13px -apple-system,Helvetica,Arial,sans-serif;overflow:hidden}
 #info{position:absolute;top:12px;left:12px;background:rgba(0,0,0,.55);padding:10px 12px;border-radius:8px;line-height:1.5}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
</style>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
            "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"}}
</script></head>
<body><div id="info">__INFO__</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

function bytes(b64){const s=atob(b64);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
const pos = new Float32Array(bytes("__POS__").buffer);
const col = new Uint8Array(bytes("__COL__").buffer);
const idxRest = new Uint32Array(bytes("__IDX_REST__").buffer);
const idxCeil = new Uint32Array(bytes("__IDX_CEIL__").buffer);
const BB = __BBOX__;

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15181d);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(3, 6, 4); scene.add(sun);

const posAttr = new THREE.BufferAttribute(pos, 3);
const colAttr = new THREE.BufferAttribute(col, 3, true);
function makeMesh(index){
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', posAttr);
  g.setAttribute('color', colAttr);
  g.setIndex(new THREE.BufferAttribute(index, 1));
  g.computeVertexNormals();
  return new THREE.Mesh(g, new THREE.MeshLambertMaterial({vertexColors:true, side:THREE.DoubleSide}));
}
const rest = makeMesh(idxRest); scene.add(rest);
const ceil = makeMesh(idxCeil); ceil.visible = false; scene.add(ceil);

const center = new THREE.Vector3((BB.min[0]+BB.max[0])/2, (BB.min[1]+BB.max[1])/2, (BB.min[2]+BB.max[2])/2);
const size = Math.max(BB.max[0]-BB.min[0], BB.max[1]-BB.min[1], BB.max[2]-BB.min[2]);
const grid = new THREE.GridHelper(Math.ceil(size)+2, Math.ceil(size)+2, 0x555555, 0x333333);
grid.position.set(center.x, BB.min[1]-0.01, center.z); scene.add(grid);
scene.add(new THREE.AxesHelper(0.5)); // начало системы W: X красный, Y зелёный (вверх), Z синий

const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.01, 200);
camera.position.set(center.x + size*0.9, center.y + size*1.1, center.z + size*0.9);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(center); controls.update();

window.addEventListener('resize', ()=>{camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth, window.innerHeight);});
window.addEventListener('keydown', e=>{
  if(e.key==='c'||e.key==='C'){ceil.visible=!ceil.visible;}
  if(e.key==='w'||e.key==='W'){rest.material.wireframe=!rest.material.wireframe;ceil.material.wireframe=rest.material.wireframe;}
});
// __EXTRA_JS__
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
</script></body></html>
"""


def build_viewer_html(session: Path, extra_js: str = "", extra_info: str = "", verbose: bool = True,
                      vertex_colors=None, legend_html=None) -> str:
    """HTML-просмотрщик меша сессии. extra_js выполняется в том же модуле (доступны THREE и scene).
    vertex_colors — свои цвета вершин (N×3 uint8) вместо цветов классов; legend_html — своя легенда."""
    ply = session / "scan" / "mesh.ply"
    if not ply.exists():
        sys.exit(f"Не найден {ply}")

    verts, idx, cls = read_ply(ply)
    nv, nf = len(verts), len(idx)
    bmin, bmax = verts.min(axis=0), verts.max(axis=0)
    dims = bmax - bmin

    # Цвет вершины = цвет класса последнего треугольника, который её использует.
    palette = np.array([CLASSES.get(i, CLASSES[0])[1] for i in range(256)], dtype=np.uint8)
    vcol = np.zeros((nv, 3), dtype=np.uint8)
    vcol[idx.reshape(-1)] = np.repeat(palette[cls], 3, axis=0)
    if vertex_colors is not None:
        vcol = np.ascontiguousarray(vertex_colors, dtype=np.uint8)

    ceil_mask = cls == 3
    idx_rest = idx[~ceil_mask].reshape(-1)
    idx_ceil = idx[ceil_mask].reshape(-1)

    counts = {CLASSES[k][0]: int(v) for k, v in zip(*np.unique(cls, return_counts=True)) if k in CLASSES}
    if verbose:
        print(f"Сессия:        {session.name}")
        print(f"Вершины:       {nv:,}")
        print(f"Треугольники:  {nf:,}")
        print(f"Размеры (м):   X {dims[0]:.2f}  ×  Y (высота) {dims[1]:.2f}  ×  Z {dims[2]:.2f}")
        print("Классы:        " + ", ".join(f"{k}: {v:,}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))

    legend = legend_html if legend_html is not None else "".join(
        f'<span class="sw" style="background:rgb{CLASSES[k][1]}"></span>{CLASSES[k][0]} ({counts.get(CLASSES[k][0], 0):,})<br>'
        for k in sorted(CLASSES))
    info = (f"<b>{session.name}</b><br>{nv:,} вершин · {nf:,} треугольников<br>"
            f"{dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} м (X × высота × Z)<br><br>{legend}<br>"
            f"{extra_info}"
            "мышь: вращать / колесо / правая кнопка<br>C — потолок · W — каркас")

    return (HTML.replace("__TITLE__", session.name)
                .replace("__INFO__", info)
                .replace("__POS__", b64(verts.astype("<f4")))
                .replace("__COL__", b64(vcol))
                .replace("__IDX_REST__", b64(idx_rest.astype("<u4")))
                .replace("__IDX_CEIL__", b64(idx_ceil.astype("<u4")))
                .replace("__BBOX__", json.dumps({"min": bmin.tolist(), "max": bmax.tolist()}))
                .replace("// __EXTRA_JS__", extra_js))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="папка сессии, например data/sessions/20261003-201659-scan")
    ap.add_argument("--no-open", action="store_true", help="не открывать браузер")
    args = ap.parse_args()

    session = Path(args.session)
    html = build_viewer_html(session)
    out = session / "scan" / "mesh_view.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nПросмотр:      {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()