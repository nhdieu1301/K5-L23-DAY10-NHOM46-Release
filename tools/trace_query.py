"""Script truy vet ticket team AI (Pha 5 & De an) -> xuat trace.json.

Tieu chi:
- Lay phien ban guideline theo job trong jobs.csv (khong lay o metadata anh).
- tier_a: anh thuoc scene team AI neu, job dung guideline cu, co it nhat 1 box detection thuoc lop bi doi quy tac.
- tier_b: anh khac cung thuoc job guideline cu va co box detection lop do, nhung ngoai cac scene o tier_a.
- notify: tap hop ma annotator va reviewer cua cac job chua anh o tier_a va tier_b.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def load_classes(classes_path: Path) -> dict[str, int]:
    name_to_id: dict[str, int] = {}
    for line in classes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            name_to_id[parts[1].strip()] = int(parts[0])
    return name_to_id


def load_jobs(jobs_path: Path) -> dict[str, dict[str, str]]:
    jobs: dict[str, dict[str, str]] = {}
    with jobs_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            jid = row["job_id"].strip()
            jobs[jid] = {
                "annotator": row["annotator"].strip(),
                "reviewer": row["reviewer"].strip(),
                "guideline": row["guideline"].strip(),
            }
    return jobs


def load_metadata(meta_path: Path) -> dict[str, dict]:
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    items: list = []
    if isinstance(raw, dict):
        if isinstance(raw.get("items"), list):
            items = raw["items"]
        elif isinstance(raw.get("images"), list):
            items = raw["images"]
        else:
            return {Path(str(k)).stem: v for k, v in raw.items() if isinstance(v, dict)}
    elif isinstance(raw, list):
        items = raw

    result: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get("image") or item.get("file_name") or item.get("id") or ""
        stem = Path(str(key)).stem
        if stem:
            result[stem] = item
    return result


def detect_target_class(changelog_text: str, ticket_text: str, valid_classes: set[str]) -> str:
    for text in (changelog_text, ticket_text):
        for token in re.findall(r"`([^`]+)`", text):
            token = token.strip()
            if token in valid_classes:
                return token
    for text in (changelog_text, ticket_text):
        for cname in sorted(valid_classes, key=len, reverse=True):
            if re.search(rf"\b{re.escape(cname)}\b", text):
                return cname
    raise ValueError("Khong tu dong tim thay ten lop trong changelog.md / ticket.md. Hay truyen --class-name.")


def detect_old_guideline(changelog_text: str, jobs: dict[str, dict[str, str]]) -> str:
    arrow_match = re.search(r"\b(v[0-9.]+)\s*(?:->|→)\s*v[0-9.]+", changelog_text)
    if arrow_match:
        return arrow_match.group(1).strip()

    versions = sorted({info["guideline"] for info in jobs.values() if info.get("guideline")})
    if versions:
        return versions[0]

    raise ValueError("Khong xac dinh duoc guideline cu. Hay truyen --old-guideline.")


def detect_reported_scenes(ticket_text: str) -> set[str]:
    return set(re.findall(r"\bscene_[A-Za-z0-9_-]+\b", ticket_text))


def image_has_class(label_path: Path, target_class_id: int) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0].isdigit() and int(parts[0]) == target_class_id:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Truy vet ticket team AI va xuat trace.json")
    parser.add_argument("delivery", type=Path, help="Thu muc delivery (vd: runs/r46007/delivery)")
    parser.add_argument("--out", type=Path, required=True, help="Duong dan file trace.json dau ra")
    parser.add_argument("--class-name", type=str, default="", help="Ghi de ten lop bi doi quy tac (vd: cup)")
    parser.add_argument("--old-guideline", type=str, default="", help="Ghi de phien ban guideline cu (vd: v1.2)")
    parser.add_argument("--scenes", type=str, default="", help="Ghi de danh sach scene cach nhau dau phay")
    args = parser.parse_args()

    name_to_id = load_classes(args.delivery / "classes.txt")
    jobs = load_jobs(args.delivery / "jobs.csv")
    metadata = load_metadata(args.delivery / "metadata.json")

    changelog_path = args.delivery / "guideline" / "changelog.md"
    ticket_path = args.delivery / "incoming" / "ticket.md"
    changelog_text = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
    ticket_text = ticket_path.read_text(encoding="utf-8") if ticket_path.exists() else ""

    target_class = args.class_name.strip() or detect_target_class(changelog_text, ticket_text, set(name_to_id.keys()))
    if target_class not in name_to_id:
        raise ValueError(f"Lop '{target_class}' khong co trong classes.txt")
    target_class_id = name_to_id[target_class]

    old_guideline = args.old_guideline.strip() or detect_old_guideline(changelog_text, jobs)
    reported_scenes = (
        {s.strip() for s in args.scenes.split(",") if s.strip()}
        if args.scenes.strip()
        else detect_reported_scenes(ticket_text)
    )

    print(f"[Trace] Lop doi quy tac : {target_class} (id={target_class_id})")
    print(f"[Trace] Guideline cu    : {old_guideline}")
    print(f"[Trace] Scene bao loi   : {sorted(reported_scenes)}")

    tier_a: list[str] = []
    tier_b: list[str] = []
    notify: set[str] = set()

    for stem in sorted(metadata.keys()):
        info = metadata[stem]
        job_id = str(info.get("job_id", "")).strip()
        scene_id = str(info.get("scene_id", "")).strip()
        job = jobs.get(job_id)
        if not job:
            continue
        if job["guideline"] != old_guideline:
            continue

        label_path = args.delivery / "labels" / f"{stem}.txt"
        if not image_has_class(label_path, target_class_id):
            continue

        if scene_id in reported_scenes:
            tier_a.append(stem)
        else:
            tier_b.append(stem)

        if job["annotator"]:
            notify.add(job["annotator"])
        if job["reviewer"]:
            notify.add(job["reviewer"])

    output = {
        "tier_a": sorted(tier_a),
        "tier_b": sorted(tier_b),
        "notify": sorted(notify),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Trace] Da ghi {args.out}: tier_a={len(tier_a)}, tier_b={len(tier_b)}, notify={output['notify']}")


if __name__ == "__main__":
    main()