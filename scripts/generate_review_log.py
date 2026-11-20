#!/usr/bin/env python3
import json
import csv
import argparse
from pathlib import Path

def build_review_log(candidates_path: Path, csv_path: Path, reviewers: list):
    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    fieldnames = ["image_id", "error_type", "bbox", "is_true_error", "reviewer", "notes"]
    
    num_reviewers = len(reviewers)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, cand in enumerate(candidates):
            assigned_reviewer = reviewers[idx % num_reviewers] if reviewers else "Dev2"
            writer.writerow({
                "image_id": cand.get("image_id"),
                "error_type": cand.get("error_type"),
                "bbox": str(cand.get("bbox")),
                "is_true_error": 1,  # Mặc định nghi vấn là 1, reviewer sửa thành 0 nếu nhiễu
                "reviewer": assigned_reviewer,
                "notes": cand.get("details", "")
            })

    print(f"[+] Đã tạo file {csv_path} với {len(candidates)} bản ghi chia cho: {reviewers}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xuất review_log.csv cho team")
    parser.add_argument("--candidates", type=Path, default=Path("candidates.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("review_log.csv"))
    parser.add_argument("--members", type=str, default="Dev1,Dev2,Dev3", help="Danh sách tên thành viên phân chia, phân tách dấu phẩy")
    args = parser.parse_args()

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    build_review_log(args.candidates, args.out-csv, members)