# Dùng AI

| # | Ai dùng | Công cụ | Prompt (tóm tắt) | AI sai ở đâu | Nhóm kiểm lại thế nào |
|---|---|---|---|---|---|
| 1 | Nguyễn Minh Hiếu | Antigravity AI | Viết phần kiểm tra 9 loại lỗi luật trong `validator/validate.py` (`missing_image`, `wrong_hash`, `invalid_class`, `bbox_out_of_bounds`, `duplicate_box`, `scene_leak`, `guideline_mismatch`, `invalid_category_id`, `thing_overlap`) chỉ dùng thư viện chuẩn và `scripts/pngutil.py`. | Ban đầu nếu chỉ dùng ngưỡng IoU cho `duplicate_box` dễ dính cặp box chồng tự nhiên trong COCO; ngoài ra khi kiểm `wrong_hash` cần bỏ qua file đã bị `missing_image` để không báo trùng 2 lỗi trên cùng 1 ảnh. | Kiểm tra kích thước lệch `<= 1px` kết hợp độ lệch tâm `<= 3px` và `IoU > 0.7`; chạy thử trên `runs/r46007`, seed phụ `runs/r46008` và lô sạch `runs/clean` (`--no-faults`), đối chiếu bằng `score_findings.py` đạt F1 lỗi luật ≈ 1.0 (0 lỗi báo nhầm trên lô sạch). |
