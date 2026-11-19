"""Cong cu ve chong nhan Detection (YOLO) va vien Mask Panoptic de review anh.

Chi dung cho viec xem anh bang mat (Pha 2 & De an), khong dung trong validator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_classes(classes_path: Path) -> dict[int, str]:
    classes: dict[int, str] = {}
    if not classes_path.exists():
        return classes
    for line in classes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            classes[int(parts[0])] = parts[1]
    return classes


def build_panoptic_index(panoptic_path: Path) -> tuple[dict[int, dict], dict[str, list[dict]]]:
    if not panoptic_path.exists():
        return {}, {}
    data = json.loads(panoptic_path.read_text(encoding="utf-8"))
    categories = {int(c["id"]): c for c in data.get("categories", [])}
    ann_by_stem: dict[str, list[dict]] = {}
    for ann in data.get("annotations", []):
        file_name = ann.get("file_name", "")
        stem = Path(file_name).stem if file_name else str(ann.get("image_id", ""))
        if stem:
            ann_by_stem[stem] = ann.get("segments_info", [])
    return categories, ann_by_stem


def draw_tag(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, bg_color: tuple[int, int, int], font) -> None:
    bbox = draw.textbbox((x, y), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    ty = max(0, y - th - 4)
    draw.rectangle([x, ty, x + tw + 6, ty + th + 4], fill=bg_color)
    draw.text((x + 3, ty + 2), text, fill=(0, 0, 0), font=font)


def render_image(
    delivery_dir: Path,
    stem: str,
    out_dir: Path,
    classes: dict[int, str],
    categories: dict[int, dict],
    ann_by_stem: dict[str, list[dict]],
    things_only: bool = False,
    min_side: int = 640,
    note_header: str = "",
) -> Path | None:
    img_path = delivery_dir / "images" / f"{stem}.jpg"
    if not img_path.exists():
        print(f"[BO QUA] Thieu file anh: {img_path}")
        return None

    base = Image.open(img_path).convert("RGB")
    w, h = base.size

    # Tu dong phong to anh nho (vd 240x320) de nhin ro chi tiet khi review
    scale = 1.0
    if max(w, h) < min_side:
        scale = round(min_side / max(w, h), 2)
        new_w, new_h = int(w * scale), int(h * scale)
        base = base.resize((new_w, new_h), Image.Resampling.BILINEAR)
    else:
        new_w, new_h = w, h

    overlay = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    # 1. Ve vien Mask Panoptic (mau xanh cyan cho thing, vang nhat cho stuff)
    mask_path = delivery_dir / "masks" / f"{stem}.png"
    segments = ann_by_stem.get(stem, [])
    seg_meta = {int(s["id"]): s for s in segments}

    if mask_path.exists():
        mask_img = Image.open(mask_path).convert("RGB")
        if (new_w, new_h) != (w, h):
            mask_img = mask_img.resize((new_w, new_h), Image.Resampling.NEAREST)
        raw = mask_img.tobytes()
        seg_ids = [raw[i] + (raw[i + 1] << 8) + (raw[i + 2] << 16) for i in range(0, len(raw), 3)]

        # Tim diem bien cua tung segment
        border_px = overlay.load()
        for y in range(new_h):
            row_off = y * new_w
            for x in range(new_w):
                sid = seg_ids[row_off + x]
                if sid == 0:
                    continue
                info = seg_meta.get(sid)
                cat = categories.get(int(info["category_id"]), {}) if info else {}
                is_thing = bool(cat.get("isthing", 1))
                if things_only and not is_thing:
                    continue

                is_edge = (
                    x == 0
                    or x == new_w - 1
                    or y == 0
                    or y == new_h - 1
                    or seg_ids[row_off + x - 1] != sid
                    or seg_ids[row_off + x + 1] != sid
                    or seg_ids[row_off - new_w + x] != sid
                    or seg_ids[row_off + new_w + x] != sid
                )
                if is_edge:
                    border_px[x, y] = (0, 255, 255, 255) if is_thing else (255, 220, 0, 180)

        # Ghi nhan ten lop Panoptic tai goc bbox segment
        for s in segments:
            cat = categories.get(int(s.get("category_id", -1)), {})
            is_thing = bool(cat.get("isthing", 1))
            if things_only and not is_thing:
                continue
            bx, by, _, _ = s.get("bbox", [0, 0, 0, 0])
            sx, sy = int(bx * scale), int(by * scale)
            cname = cat.get("name", f"cat_{s.get('category_id')}")
            crowd_tag = " [crowd]" if s.get("iscrowd") else ""
            tag_color = (0, 230, 230) if is_thing else (230, 200, 0)
            draw_tag(draw, sx, sy + 14, f"PAN: {cname}{crowd_tag}", tag_color, font)

    # 2. Ve BBox Detection YOLO (mau do)
    label_path = delivery_dir / "labels" / f"{stem}.txt"
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            x1 = int((xc - bw / 2.0) * new_w)
            y1 = int((yc - bh / 2.0) * new_h)
            x2 = int((xc + bw / 2.0) * new_w)
            y2 = int((yc + bh / 2.0) * new_h)

            draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50, 255), width=2)
            cname = classes.get(cid, f"INVALID_{cid}")
            draw_tag(draw, max(0, x1), max(0, y1), f"DET: {cname}", (255, 70, 70), font)

    if note_header:
        draw.rectangle([0, 0, new_w, 18], fill=(0, 0, 0, 200))
        draw.text((4, 3), note_header, fill=(255, 255, 255), font=font)

    combined = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_overlay.jpg"
    combined.save(out_path, quality=92)
    return out_path


def extract_candidate_items(cand_path: Path) -> list[tuple[str, str]]:
    raw = json.loads(cand_path.read_text(encoding="utf-8"))
    items = raw.get("candidates", raw) if isinstance(raw, dict) else raw
    result: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, dict):
            stem = str(item.get("image") or item.get("target") or "")
            ctype = str(item.get("type") or "")
            if stem:
                result.append((stem, ctype))
        elif isinstance(item, str):
            result.append((item, ""))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ve overlay detection + panoptic de review anh")
    parser.add_argument("delivery", type=Path, help="Thu muc delivery (vd: runs/r46007/delivery)")
    parser.add_argument("--out", type=Path, default=Path("runs/preview"), help="Thu muc xuat anh preview")
    parser.add_argument("--image", type=str, default="", help="Ten 1 anh can xem (khong duoi, vd: 000000000139)")
    parser.add_argument("--candidates", type=Path, default=None, help="File candidates.json de xuat hang loat")
    parser.add_argument("--start", type=int, default=0, help="Index bat dau trong candidates")
    parser.add_argument("--end", type=int, default=None, help="Index ket thuc (khong bao gom) trong candidates")
    parser.add_argument("--things-only", action="store_true", help="Chi ve vien segment thing (bo qua stuff)")
    args = parser.parse_args()

    classes = load_classes(args.delivery / "classes.txt")
    categories, ann_by_stem = build_panoptic_index(args.delivery / "panoptic.json")

    if args.image:
        out = render_image(
            args.delivery,
            args.image,
            args.out,
            classes,
            categories,
            ann_by_stem,
            things_only=args.things_only,
            note_header=f"Image: {args.image} | DO=Detection | XANH=Panoptic",
        )
        if out:
            print(f"Da luu: {out}")
        return

    if args.candidates:
        cands = extract_candidate_items(args.candidates)
        subset = cands[args.start : args.end]
        for idx, (stem, ctype) in enumerate(subset, start=args.start):
            header = f"[{idx}] {stem} | Goi y: {ctype} | DO=Det | XANH=Panoptic"
            out = render_image(
                args.delivery,
                stem,
                args.out,
                classes,
                categories,
                ann_by_stem,
                things_only=args.things_only,
                note_header=header,
            )
            if out:
                print(f"[{idx}] -> {out}")
        return

    # Mac dinh neu khong truyen --image hay --candidates: xuat thu 5 anh dau tien de test
    sample_imgs = sorted((args.delivery / "images").glob("*.jpg"))[:5]
    for img_file in sample_imgs:
        out = render_image(
            args.delivery,
            img_file.stem,
            args.out,
            classes,
            categories,
            ann_by_stem,
            things_only=args.things_only,
            note_header=f"Test: {img_file.stem} | DO=Det | XANH=Panoptic",
        )
        if out:
            print(f"Da luu: {out}")


if __name__ == "__main__":
    main()