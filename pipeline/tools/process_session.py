"""
AttentionScape — вся обработка сессии одной командой.

Запуск из корня репозитория:
    python pipeline/tools/process_session.py                 # самый свежий zip сессии в ~/Downloads
    python pipeline/tools/process_session.py путь/к/сессии.zip
    python pipeline/tools/process_session.py data/sessions/<session_id>   # уже распакованная сессия

Шаги:
    1) zip с iPhone (кнопка Share) распаковывается в data/sessions/<session_id>/;
       старые результаты этой сессии (perception/, results/) удаляются — они считаются заново;
    2) detect_people.py — люди и положение головы в 3D;
    3) gaze_attention.py — направление взгляда и карта внимания (results/);
    4) attention.json и attention_f32.bin копируются в iCloud Drive → AttentionScape/<session_id>/,
       откуда их забирает кнопка Import map на iPhone;
    5) view_attention.py — 3D-просмотр в браузере (отключается --no-view).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
SESSIONS = REPO / "data" / "sessions"
ICLOUD = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
SESSION_ZIP = re.compile(r"^\d{8}-\d{6}-.+\.zip$")


def newest_download() -> Path:
    downloads = Path.home() / "Downloads"
    zips = [p for p in downloads.glob("*.zip") if SESSION_ZIP.match(p.name)]
    if not zips:
        sys.exit(f"В {downloads} нет zip сессий (имя вида 20261017-143000-scan-....zip). Укажите путь явно.")
    return max(zips, key=lambda p: p.stat().st_mtime)


def find_session_root(folder: Path) -> Path:
    """Папка, где лежит session.json (в zip с iPhone она обычно вложена на один уровень)."""
    candidates = sorted(folder.rglob("session.json"), key=lambda p: len(p.parts))
    if not candidates:
        sys.exit("В архиве нет session.json — это не сессия AttentionScape")
    return candidates[0].parent


def unpack(zip_path: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                target = (tmp / name).resolve()
                if not target.is_relative_to(tmp.resolve()):
                    sys.exit(f"Подозрительный путь в архиве: {name}")
            archive.extractall(tmp)
        root = find_session_root(tmp)
        session_id = json.loads((root / "session.json").read_text(encoding="utf-8"))["session_id"]
        dest = SESSIONS / session_id
        if dest.exists():
            shutil.rmtree(dest)          # сессия с iPhone полнее: в ней все кадры, включая старые
        SESSIONS.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, dest)
    for junk in dest.rglob("__MACOSX"):
        shutil.rmtree(junk, ignore_errors=True)
    return dest


def run(script: str, *args: str) -> None:
    print(f"\n=== {script} ===", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, str(TOOLS / script), *args], cwd=REPO)
    if result.returncode != 0:
        sys.exit(f"{script} завершился с ошибкой (код {result.returncode}) — обработка остановлена")
    print(f"--- {script}: {time.time() - started:.0f} с", flush=True)


def copy_to_icloud(session: Path) -> Path | None:
    if not ICLOUD.exists():
        print("\niCloud Drive не найден на этом Mac — перенесите results/ на iPhone вручную")
        return None
    dest = ICLOUD / "AttentionScape" / session.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("attention.json", "attention_f32.bin"):
        shutil.copy2(session / "results" / name, dest / name)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", help="zip сессии или папка data/sessions/<id> (по умолчанию — свежий zip в ~/Downloads)")
    ap.add_argument("--no-view", action="store_true", help="не открывать 3D-просмотр")
    ap.add_argument("--no-icloud", action="store_true", help="не копировать карту в iCloud Drive")
    ap.add_argument("--no-smooth", action="store_true", help="не сглаживать взгляд по треку")
    args = ap.parse_args()

    source = Path(args.source).expanduser() if args.source else newest_download()
    if not source.exists():
        sys.exit(f"Не найдено: {source}")
    started = time.time()

    if source.is_file() and source.suffix == ".zip":
        print(f"Архив: {source}")
        session = unpack(source)
    elif (source / "session.json").exists():
        session = source.resolve()
    else:
        sys.exit("Нужен zip сессии с iPhone или папка сессии с session.json")
    for old in ("perception", "results", "check"):
        shutil.rmtree(session / old, ignore_errors=True)

    frames = len(list((session / "capture" / "frames").glob("*.json")))
    if frames == 0:
        sys.exit(f"В сессии {session.name} нет записанных кадров — сначала Record на iPhone")
    print(f"Сессия: {session.name}, кадров: {frames}")

    session_arg = str(session.relative_to(REPO)) if session.is_relative_to(REPO) else str(session)
    run("detect_people.py", session_arg)
    run("gaze_attention.py", session_arg, *(["--no-smooth"] if args.no_smooth else []))

    icloud = None if args.no_icloud else copy_to_icloud(session)
    if not args.no_view:
        run("view_attention.py", session_arg)

    print(f"\nГотово за {time.time() - started:.0f} с. Сессия: {session}")
    if icloud:
        print(f"Карта для iPhone: iCloud Drive → AttentionScape → {session.name} "
              f"(Import map → выбрать папку {session.name} или оба файла)")


if __name__ == "__main__":
    main()