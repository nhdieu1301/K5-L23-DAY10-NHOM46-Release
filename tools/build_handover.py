#!/usr/bin/env python3
"""Pha 4 — dựng lô bàn giao v2 (detection) từ delivery v1 + errors.json, KHÔNG sửa delivery/.

  python tools/build_handover.py --run runs/r46007            # dựng handover/<rid>_v2/
  python tools/build_handover.py --run runs/r46007 --pack --lineage
  python scripts/check_handover.py --run runs/r46007

Luật (GUIDE.md Pha 4, scripts/check_handover.py):
- Lỗi nhãn detection / thiếu file (DET_REWORK) -> rework_queue.csv, ảnh rời v2.
- wrong_hash: mở ảnh kiểm; ảnh còn đọc được -> giữ, ghi hash thật vào manifest.csv; ảnh hỏng -> rework.
- guideline_mismatch: truyvet.csv lấy guideline theo job (jobs.csv).
- scene_leak: dồn cả cảnh về một split (split chiếm đa số, hoà thì train).
- Lỗi chỉ ở panoptic: ảnh ở lại v2, liệt kê trong README Ghi chú.
Nhãn trong v2 được copy byte-y-nguyên từ v1 (không sửa tay).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DET_REWORK = {"missing_image", "invalid_class", "bbox_out_of_bounds", "duplicate_box", "class_swap", "bbox_drift", "missing_box"}
PANOPTIC_ONLY = {"invalid_category_id", "thing_overlap", "category_swap", "instance_merge"}
THRESHOLD = 0.02

ACTION = {
    "missing_image": "bổ sung lại file ảnh rồi kiểm nhãn",
    "invalid_class": "gán lại class_id hợp lệ (0-79)",
    "bbox_out_of_bounds": "vẽ lại box nằm trong ảnh",
    "duplicate_box": "xoá box trùng, giữ một box",
    "class_swap": "gán lại đúng lớp theo guideline của job",
    "bbox_drift": "vẽ lại box khít vật thể",
    "missing_box": "bổ sung box còn thiếu",
    "wrong_hash": "ảnh hỏng: lấy lại ảnh gốc rồi kiểm nhãn",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, header: list[str], rows: list[list]) -> None:
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def image_ok(p: Path) -> bool:
    """Ảnh còn nguyên? Ưu tiên Pillow (decode thật), không có thì kiểm marker JPEG."""
    if not p.is_file():
        return False
    try:
        from PIL import Image
        with Image.open(p) as im:
            im.load()
        return True
    except ImportError:
        data = p.read_bytes()
        return data.startswith(b"\xff\xd8") and data.rstrip(b"\x00").endswith(b"\xff\xd9")
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Dựng lô bàn giao v2")
    ap.add_argument("--run", required=True)
    ap.add_argument("--errors", help="mặc định <run>/errors.json")
    ap.add_argument("--v2-decision", default="Release", choices=["Release", "Hold", "Rework"])
    ap.add_argument("--pack", action="store_true", help="đóng gói zip + .sha256 (pack_release)")
    ap.add_argument("--lineage", action="store_true", help="ghi HOLD v1 (nếu chưa có) và release v2 vào <run>/../lineage.jsonl")
    args = ap.parse_args()

    run = Path(args.run)
    d = run / "delivery"
    rid = json.loads((d / "release.json").read_text(encoding="utf-8"))["release_id"]
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["items"]
    meta = {m["image"]: m for m in json.loads((d / "metadata.json").read_text(encoding="utf-8"))["items"]}
    jobs = {r["job_id"]: r for r in read_csv(d / "jobs.csv")}
    names80 = [ln.split(" ", 1)[1].strip() for ln in (d / "classes.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    errors = json.loads(Path(args.errors or run / "errors.json").read_text(encoding="utf-8"))["errors"]

    by_img: dict[str, list[str]] = defaultdict(list)
    for e in errors:
        by_img[e["target"]].append(e["type"])

    # 1) Quyết định rework
    rework: dict[str, str] = {}
    hash_fixed, panoptic_imgs = [], []
    for name in man:
        types = by_img.get(name, [])
        det = [t for t in types if t in DET_REWORK]
        if det:
            rework[name] = det[0]
        elif "wrong_hash" in types:
            if image_ok(d / man[name]["image"]):
                hash_fixed.append(name)
            else:
                rework[name] = "wrong_hash"
        elif not (d / man[name]["image"]).is_file():
            rework[name] = "missing_image"
        if name not in rework and any(t in PANOPTIC_ONLY for t in types):
            panoptic_imgs.append(name)
    keep = sorted(n for n in man if n not in rework)

    # 2) Sửa scene_leak: mỗi cảnh một split (đa số trên toàn cảnh v1, hoà -> train)
    scene_split = {}
    by_scene = defaultdict(list)
    for n, m in meta.items():
        by_scene[m["scene_id"]].append(m["split"])
    for sc, splits in by_scene.items():
        c = Counter(splits)
        scene_split[sc] = sorted(c, key=lambda s: (-c[s], s != "train", s))[0]
    leaked = sorted(sc for sc, s in by_scene.items() if len(set(s)) > 1)

    # 3) Ghi lô v2 (xoá bản dựng cũ của v2, không đụng delivery/)
    v2 = run / "handover" / f"{rid}_v2"
    if v2.exists():
        shutil.rmtree(v2)
    (v2 / "images").mkdir(parents=True)
    (v2 / "labels").mkdir()
    counts: Counter = Counter()
    for n in keep:
        it = man[n]
        img, lab = d / it["image"], d / it["label"]
        shutil.copyfile(img, v2 / "images" / f"{n}{img.suffix}")
        shutil.copyfile(lab, v2 / "labels" / f"{n}.txt")
        for ln in lab.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                c = int(float(ln.split()[0]))
                counts[names80[c] if 0 <= c < len(names80) else str(c)] += 1

    files = sorted(p.relative_to(v2).as_posix() for sub in ("images", "labels") for p in (v2 / sub).iterdir() if p.is_file())
    write_csv(v2 / "manifest.csv", ["file", "sha256"], [[f, sha(v2 / f)] for f in files])
    order = {nm: i for i, nm in enumerate(names80)}
    summ = [[k, v] for k, v in sorted(counts.items(), key=lambda kv: order.get(kv[0], 999))]
    write_csv(v2 / "label_summary.csv", ["class", "count"], summ + [["TOTAL", sum(counts.values())]])

    tv = []
    for n in keep:
        m = meta[n]
        j = jobs[m["job_id"]]
        tv.append([n, m["scene_id"], scene_split[m["scene_id"]], m["job_id"], j["annotator"], j["reviewer"], j["guideline"]])
    write_csv(v2 / "truyvet.csv", ["image", "scene_id", "split", "job_id", "annotator", "reviewer", "guideline"], tv)

    rq = []
    for n in sorted(rework):
        j = jobs[meta[n]["job_id"]]
        rq.append([n, rework[n], j["annotator"], j["reviewer"], ACTION.get(rework[n], "gán lại")])
    write_csv(v2 / "rework_queue.csv", ["image", "type", "annotator", "reviewer", "action"], rq)

    # 4) README.txt
    gl = defaultdict(list)
    for jid in sorted(jobs):
        gl[jobs[jid]["guideline"]].append(jid)
    gl_txt = ", ".join(f"{v} ({js[0]}-{js[-1]})" if len(js) > 1 else f"{v} ({js[0]})" for v, js in sorted(gl.items()))
    n_err = len(by_img)
    rate = n_err / len(man)
    v1_dec = "Hold" if rate > THRESHOLD else "Release"
    gm = sorted(n for n, ts in by_img.items() if "guideline_mismatch" in ts)
    rw_people = sorted({f"{r[2]}/{r[3]}" for r in rq})
    readme = [
        f"Release: {rid}_v2",
        f"Parent: {rid}",
        f"Số ảnh: {len(keep)}",
        f"Số nhãn: {sum(counts.values())}",
        f"Guideline: {gl_txt}",
        f"QC v1: {n_err}/{len(man)} = {rate * 100:.1f}%".replace(".", ","),
        f"Kết luận v1: {v1_dec} — {n_err} ảnh/cảnh lỗi trên {len(man)} ảnh ({rate:.1%}) "
        + ("vượt ngưỡng 2%, không phát hành v1" if rate > THRESHOLD else "trong ngưỡng 2%"),
        f"Kết luận v2: {args.v2_decision} — tách {len(rework)} ảnh lỗi nhãn detection/thiếu file sang rework_queue; "
        f"sửa {len(hash_fixed)} hash, {len(gm)} guideline theo job, {len(leaked)} cảnh rò split; "
        f"còn {len(panoptic_imgs)} ảnh lỗi panoptic ở lại v2",
        "Ghi chú: "
        f"split: dồn cả cảnh về split đa số ({', '.join(leaked) or 'không có'}); "
        f"hash: đã mở ảnh kiểm, ảnh còn nguyên nên ghi hash mới ({', '.join(hash_fixed) or 'không có'}); "
        f"guideline: truyvet lấy theo jobs.csv ({', '.join(gm) or 'không có'}); "
        f"rework giao lại người gán/kiểm theo jobs.csv: {', '.join(rw_people) or 'không có'}; "
        f"báo team panoptic các ảnh lỗi panoptic: {', '.join(sorted(panoptic_imgs)) or 'không có'}",
    ]
    (v2 / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"{v2}: {len(keep)} ảnh, {sum(counts.values())} nhãn, rework {len(rework)}, QC v1 {n_err}/{len(man)} = {rate:.1%} -> {v1_dec}")

    # 5) lineage + đóng gói
    if args.lineage:
        import lineage
        root = run.parent
        ev = lineage.read(root)
        if v1_dec == "Hold" and not any(e.get("event") == "decision" and e.get("release_id") == rid and e.get("decision") == "HOLD" for e in ev):
            lineage.append(root, {"event": "decision", "release_id": rid, "decision": "HOLD", "detail": f"QC v1 {rate:.1%} > 2%"})
        if not any(e.get("event") == "release" and e.get("release_id") == f"{rid}_v2" for e in ev):
            lineage.append(root, {"event": "release", "release_id": f"{rid}_v2", "parent": rid})
        print(f"lineage: {root / 'lineage.jsonl'}")
    if args.pack:
        from pack_release import build_zip
        data = build_zip(v2)
        v2.with_suffix(".zip").write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        v2.with_suffix(".sha256").write_text(f"{digest}  {v2.name}.zip\n", encoding="utf-8")
        print(f"{v2.name}.zip sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
