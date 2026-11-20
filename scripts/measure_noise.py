#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import numpy as np

def analyze_clean_noise(candidates_path: Path):
    if not candidates_path.exists():
        print(f"[-] Không tìm thấy {candidates_path}")
        return

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    stats = {}
    ious = []
    for c in candidates:
        err_type = c["error_type"]
        stats[err_type] = stats.get(err_type, 0) + 1
        if "pixel_iou" in c:
            ious.append(c["pixel_iou"])

    print("=== KẾT QUẢ ĐO NHIỄU TỰ NHIÊN NỀN BÀI CHẠY LÔ SẠCH ===")
    for err, count in stats.items():
        print(f" - {err}: {count} candidates")

    if ious:
        ious_arr = np.array(ious)
        print(f"\nPhân bố Pixel IoU nhiễu:")
        print(f" - Trung bình (Mean): {np.mean(ious_arr):.4f}")
        print(f" - Trung vị (Median): {np.median(ious_arr):.4f}")
        print(f" - Phân vị 5% (5th Percentile): {np.percentile(ious_arr, 5):.4f}")
        print(f" - Phân vị 95% (95th Percentile): {np.percentile(ious_arr, 95):.4f}")
        print("\n=> ĐỀ XUẤT NGUỠNG LỌC NHIỄU:")
        print(f"   + Đặt --drift-min-iou = {max(0.15, float(np.percentile(ious_arr, 5))):.2f}")
        print(f"   + Đặt --drift-max-iou = {min(0.65, float(np.median(ious_arr))):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đo nhiễu lô sạch cho Dev 2")
    parser.add_argument("--candidates", type=Path, default=Path("candidates.json"))
    args = parser.parse_args()
    analyze_clean_noise(args.candidates)