#!/usr/bin/env python3
"""Validator kiểm tra 9 loại lỗi luật cho lô bàn giao (P1).

Giao diện dòng lệnh:
    python validator/validate.py <thư_mục_delivery> --out <errors.json> [--candidates <candidates.json>]

Chỉ đọc thư mục delivery được truyền vào, chỉ dùng thư viện chuẩn Python và scripts/pngutil.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import pngutil  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_yolo_line(line: str) -> tuple[str, float, float, float, float] | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        return parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None


def bbox_iou_norm(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw / 2.0, bx + bw / 2.0) - max(ax - aw / 2.0, bx - bw / 2.0))
    iy = max(0.0, min(ay + ah / 2.0, by + bh / 2.0) - max(ay - ah / 2.0, by - bh / 2.0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-12)


def check_rules(delivery_dir: Path) -> list[dict[str, str]]:
    errors: set[tuple[str, str]] = set()

    manifest_path = delivery_dir / "manifest.json"
    metadata_path = delivery_dir / "metadata.json"
    jobs_path = delivery_dir / "jobs.csv"
    classes_path = delivery_dir / "classes.txt"
    panoptic_path = delivery_dir / "panoptic.json"

    manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    num_classes = int(manifest_doc.get("num_classes", 80))
    manifest_items: dict[str, dict] = manifest_doc.get("items", {})

    # Đọc danh sách class hợp lệ từ classes.txt
    valid_classes: set[int] = set(range(num_classes))
    if classes_path.is_file():
        parsed_classes: set[int] = set()
        for line in classes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
                parsed_classes.add(int(token))
        if parsed_classes:
            valid_classes = parsed_classes

    # Đọc panoptic.json để lấy kích thước ảnh, categories, và annotations
    panoptic_doc = json.loads(panoptic_path.read_text(encoding="utf-8"))
    valid_cat_ids = {c["id"] for c in panoptic_doc.get("categories", [])}
    image_dims: dict[str, tuple[int, int]] = {}
    for im in panoptic_doc.get("images", []):
        stem = Path(im["file_name"]).stem
        image_dims[stem] = (int(im["width"]), int(im["height"]))

    # 1 & 2. Kiểm tra file: missing_image và wrong_hash
    # 3, 4, 5. Kiểm tra nhãn detection: invalid_class, bbox_out_of_bounds, duplicate_box
    eps = 1e-4
    for img_name, info in manifest_items.items():
        img_rel = info.get("image", f"images/{img_name}.jpg")
        lbl_rel = info.get("label", f"labels/{img_name}.txt")
        msk_rel = info.get("mask", f"masks/{img_name}.png")

        img_file = delivery_dir / img_rel
        lbl_file = delivery_dir / lbl_rel
        msk_file = delivery_dir / msk_rel

        if not img_file.is_file():
            errors.add(("missing_image", img_name))
        else:
            expected_img_hash = info.get("image_sha256", "")
            if not expected_img_hash or sha256_file(img_file) != expected_img_hash:
                errors.add(("wrong_hash", img_name))

        if lbl_file.is_file():
            expected_lbl_hash = info.get("label_sha256", "")
            if expected_lbl_hash and sha256_file(lbl_file) != expected_lbl_hash:
                errors.add(("wrong_hash", img_name))

        if msk_file.is_file():
            expected_msk_hash = info.get("mask_sha256", "")
            if expected_msk_hash and sha256_file(msk_file) != expected_msk_hash:
                errors.add(("wrong_hash", img_name))

        if not lbl_file.is_file():
            continue

        width, height = image_dims.get(img_name, (640, 640))
        boxes: list[tuple[int, float, float, float, float]] = []
        raw_lines = [ln.strip() for ln in lbl_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

        for line in raw_lines:
            parsed = parse_yolo_line(line)
            if parsed is None:
                continue
            cls_str, x, y, w, h = parsed
            try:
                cls_id = int(cls_str)
            except ValueError:
                cls_id = -1

            if cls_id not in valid_classes:
                errors.add(("invalid_class", img_name))

            left = x - w / 2.0
            right = x + w / 2.0
            top = y - h / 2.0
            bottom = y + h / 2.0
            if (
                w <= 0.0
                or h <= 0.0
                or left < -eps
                or top < -eps
                or right > 1.0 + eps
                or bottom > 1.0 + eps
            ):
                errors.add(("bbox_out_of_bounds", img_name))

            boxes.append((cls_id, x, y, w, h))

        # Kiểm tra duplicate_box: cùng lớp, cùng kích thước, tâm lệch 1-2 pixel
        n_boxes = len(boxes)
        found_dup = False
        for i in range(n_boxes):
            if found_dup:
                break
            c1, x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, n_boxes):
                c2, x2, y2, w2, h2 = boxes[j]
                if c1 != c2:
                    continue
                dw_px = abs(w1 - w2) * width
                dh_px = abs(h1 - h2) * height
                dx_px = abs(x1 - x2) * width
                dy_px = abs(y1 - y2) * height
                if dw_px <= 1.0 and dh_px <= 1.0 and dx_px <= 3.0 and dy_px <= 3.0:
                    if bbox_iou_norm((x1, y1, w1, h1), (x2, y2, w2, h2)) > 0.7:
                        errors.add(("duplicate_box", img_name))
                        found_dup = True
                        break

    # 6 & 7. Kiểm tra metadata: scene_leak và guideline_mismatch
    job_guidelines: dict[str, str] = {}
    if jobs_path.is_file():
        with jobs_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                jid = row.get("job_id", "").strip()
                gl = row.get("guideline", "").strip()
                if jid:
                    job_guidelines[jid] = gl

    if metadata_path.is_file():
        meta_doc = json.loads(metadata_path.read_text(encoding="utf-8"))
        scene_splits: dict[str, set[str]] = {}
        for item in meta_doc.get("items", []):
            img_name = item.get("image", "").strip()
            scene_id = item.get("scene_id", "").strip()
            split = item.get("split", "").strip()
            job_id = item.get("job_id", "").strip()
            gl_ver = item.get("guideline_version", "").strip()

            if scene_id and split:
                scene_splits.setdefault(scene_id, set()).add(split)

            expected_gl = job_guidelines.get(job_id)
            if img_name and expected_gl and gl_ver != expected_gl:
                errors.add(("guideline_mismatch", img_name))

        for scene_id, splits in scene_splits.items():
            if len(splits) > 1:
                errors.add(("scene_leak", scene_id))

    # 8 & 9. Kiểm tra panoptic: invalid_category_id và thing_overlap
    for ann in panoptic_doc.get("annotations", []):
        mask_filename = ann.get("file_name", "")
        img_name = Path(mask_filename).stem
        segments = ann.get("segments_info", [])

        for seg in segments:
            if seg.get("category_id") not in valid_cat_ids:
                errors.add(("invalid_category_id", img_name))
                break

        mask_rel = manifest_items.get(img_name, {}).get("mask", f"masks/{mask_filename}")
        mask_path = delivery_dir / mask_rel
        if not mask_path.is_file():
            continue

        _, _, pixels = pngutil.read_rgb(mask_path)
        rgb_counts = Counter(zip(pixels[0::3], pixels[1::3], pixels[2::3]))

        for seg in segments:
            seg_id = int(seg.get("id", 0))
            if seg_id == 0:
                continue
            declared_area = int(seg.get("area", 0))
            actual_area = rgb_counts.get(pngutil.segment_rgb(seg_id), 0)
            if declared_area > actual_area:
                errors.add(("thing_overlap", img_name))
                break

    sorted_errors = sorted(errors, key=lambda x: (x[0], x[1]))
    return [{"type": err_type, "target": target} for err_type, target in sorted_errors]


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra lỗi luật trên thư mục delivery")
    parser.add_argument("delivery", help="Đường dẫn tới thư mục delivery")
    parser.add_argument("--out", required=True, help="Đường dẫn file errors.json đầu ra")
    parser.add_argument("--candidates", help="Đường dẫn file candidates.json (tùy chọn cho P2)")
    args = parser.parse_args()

    delivery_dir = Path(args.delivery)
    if not delivery_dir.is_dir():
        raise SystemExit(f"Không tìm thấy thư mục delivery: {delivery_dir}")

    rule_errors = check_rules(delivery_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"errors": rule_errors}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.candidates:
        cand_path = Path(args.candidates)
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        if not cand_path.exists():
            cand_path.write_text(
                json.dumps({"candidates": []}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    print(f"Đã kiểm tra {delivery_dir}: phát hiện {len(rule_errors)} lỗi luật -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
