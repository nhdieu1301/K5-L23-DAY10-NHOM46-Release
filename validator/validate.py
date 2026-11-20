#!/usr/bin/env python3
"""Validator kiểm tra 9 lỗi luật (P1) và sinh ứng viên đối chiếu det <-> panoptic (P2).

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
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import pngutil  # noqa: E402

COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

CLASS_SYNONYMS = {
    "couch": "sofa",
    "sofa": "couch",
    "tvmonitor": "tv",
    "tv monitor": "tv",
    "pottedplant": "potted plant",
    "plant": "potted plant",
    "diningtable": "dining table",
    "table": "dining table",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "phone": "cell phone",
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"^\d+\s*", "", str(name))
    clean = " ".join(name.lower().strip().replace("_", " ").replace("-", " ").split())
    return CLASS_SYNONYMS.get(clean, clean)


def load_class_map(delivery_dir: Path, num_classes: int = 80) -> dict[int, str]:
    classes_path = delivery_dir / "classes.txt"
    class_map: dict[int, str] = {}
    if classes_path.is_file():
        lines = [ln.strip() for ln in classes_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for idx, line in enumerate(lines):
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and (parts[0].isdigit() or (parts[0].startswith("-") and parts[0][1:].isdigit())):
                class_map[int(parts[0])] = parts[1].strip()
            else:
                class_map[idx] = line
    if not class_map:
        for idx in range(min(num_classes, len(COCO_80_CLASSES))):
            class_map[idx] = COCO_80_CLASSES[idx]
    return class_map


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


def compute_box_iou_px(box_a: list[int], box_b: list[float]) -> float:
    xa = max(float(box_a[0]), float(box_b[0]))
    ya = max(float(box_a[1]), float(box_b[1]))
    xb = min(float(box_a[2]), float(box_b[2]))
    yb = min(float(box_a[3]), float(box_b[3]))

    inter_area = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(1.0, float((box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
    area_b = max(1.0, float((box_b[2] - box_b[0]) * (box_b[3] - box_b[1])))
    return inter_area / (area_a + area_b - inter_area + 1e-6)


def validate_delivery(
    delivery_dir: Path,
    need_candidates: bool = False,
) -> tuple[list[dict[str, str]], list[dict]]:
    errors: set[tuple[str, str]] = set()
    candidates: list[dict] = []

    manifest_path = delivery_dir / "manifest.json"
    metadata_path = delivery_dir / "metadata.json"
    jobs_path = delivery_dir / "jobs.csv"
    panoptic_path = delivery_dir / "panoptic.json"

    manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    num_classes = int(manifest_doc.get("num_classes", 80))
    manifest_items: dict[str, dict] = manifest_doc.get("items", {})

    class_map = load_class_map(delivery_dir, num_classes=num_classes)
    valid_classes = set(class_map.keys())

    panoptic_doc = json.loads(panoptic_path.read_text(encoding="utf-8"))
    categories_by_id = {int(c["id"]): c for c in panoptic_doc.get("categories", [])}
    valid_cat_ids = set(categories_by_id.keys())

    image_dims: dict[str, tuple[int, int]] = {}
    for im in panoptic_doc.get("images", []):
        stem = Path(im["file_name"]).stem
        image_dims[stem] = (int(im["width"]), int(im["height"]))

    # Cache box detection theo tên ảnh để tái sử dụng ở bước đối chiếu P2
    det_boxes_by_img: dict[str, list[dict]] = {}

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
            det_boxes_by_img[img_name] = []
            continue

        width, height = image_dims.get(img_name, (640, 640))
        boxes: list[tuple[int, float, float, float, float]] = []
        px_boxes: list[dict] = []
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

            cls_name = class_map.get(cls_id, str(cls_id))
            x1 = int(max(0, math.floor(left * width)))
            y1 = int(max(0, math.floor(top * height)))
            x2 = int(min(width, math.ceil(right * width)))
            y2 = int(min(height, math.ceil(bottom * height)))
            px_boxes.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "norm_name": normalize_name(cls_name),
                "bbox": [x1, y1, x2, y2],
            })

        det_boxes_by_img[img_name] = px_boxes

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
    # Đồng thời đối chiếu det <-> panoptic sinh candidates nếu need_candidates=True
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

        img_w, img_h, pixels = pngutil.read_rgb(mask_path)
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

        if not need_candidates:
            continue

        det_boxes = det_boxes_by_img.get(img_name, [])
        row_stride = img_w * 3

        for seg in segments:
            seg_id = int(seg.get("id", 0))
            if seg_id == 0:
                continue
            cat_id = int(seg.get("category_id", -1))
            cat_info = categories_by_id.get(cat_id, {})

            # Chỉ xét segment thuộc nhóm thing và không phải crowd
            if cat_info.get("isthing", 0) != 1 or seg.get("iscrowd", 0) != 0:
                continue

            target_rgb = pngutil.segment_rgb(seg_id)
            segment_area = rgb_counts.get(target_rgb, 0)
            if segment_area < 800:
                continue

            tr, tg, tb = target_rgb
            seg_cls_name = cat_info.get("name", str(cat_id))
            seg_norm_name = normalize_name(seg_cls_name)
            sb = seg.get("bbox", [0, 0, img_w, img_h])
            seg_bbox = [float(sb[0]), float(sb[1]), float(sb[0] + sb[2]), float(sb[1] + sb[3])]

            evaluated_boxes = []
            for det in det_boxes:
                b = det["bbox"]
                ix1 = max(0, int(seg_bbox[0]), b[0])
                iy1 = max(0, int(seg_bbox[1]), b[1])
                ix2 = min(img_w, int(math.ceil(seg_bbox[2])), b[2])
                iy2 = min(img_h, int(math.ceil(seg_bbox[3])), b[3])

                inside_pixels = 0
                if ix1 < ix2 and iy1 < iy2:
                    for y in range(iy1, iy2):
                        row_offset = y * row_stride
                        start = row_offset + ix1 * 3
                        end = row_offset + ix2 * 3
                        for idx in range(start, end, 3):
                            if pixels[idx] == tr and pixels[idx + 1] == tg and pixels[idx + 2] == tb:
                                inside_pixels += 1

                inside_ratio = inside_pixels / float(segment_area)
                bbox_iou = compute_box_iou_px(b, seg_bbox)
                evaluated_boxes.append({
                    "det": det,
                    "inside_ratio": inside_ratio,
                    "bbox_iou": bbox_iou,
                })

            same_class_boxes = [eb for eb in evaluated_boxes if eb["det"]["norm_name"] == seg_norm_name]

            # 1. instance_merge: một segment bị phủ bởi >= 2 box cùng lớp
            high_overlap_same = [
                eb for eb in same_class_boxes
                if eb["inside_ratio"] >= 0.45 and eb["bbox_iou"] >= 0.35
            ]
            if len(high_overlap_same) >= 2:
                candidates.append({
                    "image": img_name,
                    "file_name": mask_filename,
                    "type": "instance_merge",
                    "segment_id": seg_id,
                    "class_name": seg_cls_name,
                    "details": f"1 segment mask bị phủ bởi {len(high_overlap_same)} BBox trùng lớp.",
                })

            # 2. Kiểm tra khớp cùng lớp hoặc bbox_drift
            if same_class_boxes:
                best_same = max(same_class_boxes, key=lambda x: x["bbox_iou"])
                if best_same["bbox_iou"] >= 0.30 or best_same["inside_ratio"] >= 0.40:
                    continue
                if 0.10 <= best_same["bbox_iou"] < 0.30:
                    candidates.append({
                        "image": img_name,
                        "file_name": mask_filename,
                        "type": "bbox_drift",
                        "segment_id": seg_id,
                        "class_name": seg_cls_name,
                        "inside_ratio": round(best_same["inside_ratio"], 4),
                        "bbox_iou": round(best_same["bbox_iou"], 4),
                        "details": "Bounding Box cùng lớp bị lệch so với mask.",
                    })
                    continue

            # 3. class_swap / category_swap: khớp vị trí nhưng khác lớp
            diff_class_boxes = [eb for eb in evaluated_boxes if eb["det"]["norm_name"] != seg_norm_name]
            if diff_class_boxes:
                best_diff = max(diff_class_boxes, key=lambda x: x["bbox_iou"])
                if best_diff["bbox_iou"] >= 0.50:
                    det = best_diff["det"]
                    candidates.append({
                        "image": img_name,
                        "file_name": mask_filename,
                        "type": "class_swap",
                        "segment_id": seg_id,
                        "det_class": det["class_name"],
                        "seg_class": seg_cls_name,
                        "bbox_iou": round(best_diff["bbox_iou"], 4),
                        "details": f"Lệch nhãn lớp: BBox ({det['class_name']}) vs Panoptic ({seg_cls_name}).",
                    })
                    continue

            # 4. missing_box: segment thing lớn không có box nào phủ tới
            max_inside = max((eb["inside_ratio"] for eb in evaluated_boxes), default=0.0)
            max_iou = max((eb["bbox_iou"] for eb in evaluated_boxes), default=0.0)
            if max_inside < 0.20 and max_iou < 0.20:
                candidates.append({
                    "image": img_name,
                    "file_name": mask_filename,
                    "type": "missing_box",
                    "segment_id": seg_id,
                    "class_name": seg_cls_name,
                    "details": "Segment mask không có Bounding Box nào phủ tới.",
                })

    sorted_errors = sorted(errors, key=lambda x: (x[0], x[1]))
    rule_errors = [{"type": err_type, "target": target} for err_type, target in sorted_errors]
    return rule_errors, candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra lỗi luật và đối chiếu trên thư mục delivery")
    parser.add_argument("delivery", help="Đường dẫn tới thư mục delivery")
    parser.add_argument("--out", required=True, help="Đường dẫn file errors.json đầu ra")
    parser.add_argument("--candidates", help="Đường dẫn file candidates.json (tùy chọn cho P2)")
    args = parser.parse_args()

    delivery_dir = Path(args.delivery)
    if not delivery_dir.is_dir():
        raise SystemExit(f"Không tìm thấy thư mục delivery: {delivery_dir}")

    rule_errors, candidates = validate_delivery(
        delivery_dir,
        need_candidates=bool(args.candidates),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"errors": rule_errors}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.candidates:
        cand_path = Path(args.candidates)
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        cand_path.write_text(
            json.dumps({"candidates": candidates}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Đã kiểm tra {delivery_dir}: {len(rule_errors)} lỗi luật -> {out_path}; "
            f"{len(candidates)} ứng viên -> {cand_path}"
        )
    else:
        print(f"Đã kiểm tra {delivery_dir}: phát hiện {len(rule_errors)} lỗi luật -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())