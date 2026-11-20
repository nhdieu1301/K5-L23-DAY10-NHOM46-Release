# Báo cáo nhóm 46

- **Seed luyện tập:** `46007` (Lô v1: `lab250_ff60d0167c45`, Lô v2: `lab250_ff60d0167c45_v2`)
- **Repository:** https://github.com/nhdieu1301/K5-L23-DAY10-NHOM46-Release

## 1. Kết quả validator & độ đo F1
- **Kết quả chạy trên lô luyện tập `r46007`:**
  - Tổng số lỗi phát hiện và xác nhận: **20 lỗi / 250 ảnh** (tỷ lệ lỗi **8.0%**).
  - Độ đo tự kiểm tra (`score_findings.py`): **F1 tổng ≈ 0.8** (nhóm **lỗi luật ≈ 1.0**, nhóm **lỗi gán nhãn ≈ 0.5**, **truy vết F1 = 1.0** trên cả `tier_a`, `tier_b`, `notify`).
- **Kết quả đo nhiễu tự nhiên trên lô sạch `runs/clean` (50 ảnh):**
  - Nhóm lỗi luật: **0 lỗi**.
  - Hàng đợi ứng viên đối chiếu (`candidates.json`): chỉ xuất hiện số ít ca lệch tự nhiên do vật thể bị che khuất một phần hoặc kích thước nhỏ trong COCO, giúp hàng đợi review của 5 thành viên gọn nhẹ, không bị quá tải.
- **Vì sao nếu báo hết toàn bộ ứng viên đối chiếu vào `errors.json` mà không qua review mắt người thì F1 sẽ giảm?**
  - Bộ đối chiếu tự động giữa Detection BBox và Panoptic Mask dùng ngưỡng hình học nên bắt cả các **sai lệch tự nhiên của COCO** (ví dụ: vật bị che khuất một phần khiến panoptic có mask nhưng detection bỏ qua, nhiều vật đứng sát nhau chồng lấn khung box bị nghi `instance_merge`, hoặc ca đã thuộc lỗi luật `invalid_class`). Nếu đưa toàn bộ ứng viên vào `errors.json` mà không lọc `rejected`, số lượng **Dương tính giả (False Positives)** sẽ tăng vọt, kéo **Precision** giảm mạnh và làm **F1 tổng** sụt giảm.

## 2. Quy trình rà soát ứng viên (`review_log.csv`)
- **Tổng số ứng viên sinh ra trên `r46007`:** 37 mục (tương ứng **25 cặp `(image, type)` duy nhất**).
- **Kết quả phân loại của 5 thành viên:**
  - **`confirmed` (8 lỗi thật):** `000000016451` (`category_swap`), `000000011149` (`class_swap`), `000000025424` (`class_swap`), `000000010363` (`missing_box`), và 4 ca `instance_merge` (`000000000776`, `000000013923`, `000000017182`, `000000023937`).
  - **`rejected` (17 ca loại bỏ):** 16 ca nhiễu tự nhiên (các đối tượng đứng sát/chồng khung nhưng đều có box riêng, hoặc ô tô nhỏ bị che khuất dưới gầm cầu vượt) và 1 ca (`000000001490`) có nhãn `INVALID_80` đã thuộc lỗi luật `invalid_class`.

## 3. Quyết định và lineage
- **Quyết định đối với lô v1 (`lab250_ff60d0167c45`):** **HOLD** — vì tỷ lệ lỗi thực tế là `20/250 = 8.0%`, vượt xa ngưỡng cho phép `2%`.
- **Tác động tới training run `R-021` (`lineage.py impact`):** `R-021: affected=True action=review_or_retrain`. Vì mô hình `R-021` đã huấn luyện trực tiếp trên lô `lab250_ff60d0167c45` chứa 8% dữ liệu nhiễm lỗi nhãn và siêu dữ liệu, trọng số mô hình có nguy cơ học sai đặc trưng lớp; do đó bắt buộc phải rà soát đánh giá lại (`review`) hoặc huấn luyện lại (`retrain`) trên lô sạch `lab250_ff60d0167c45_v2` (242 ảnh sạch sau khi loại 8 ảnh lỗi detection/thiếu file vào `rework_queue.csv`).

## 4. Hash chứng minh gì, không chứng minh gì?
- **SHA-256 chứng minh được:** Tính toàn vẹn dữ liệu ở mức từng byte (xác nhận file ảnh/nhãn không bị hỏng, cắt cụt, thất lạc hoặc bị sửa đổi trái phép kể từ thời điểm tính `manifest.csv` và đóng gói `.sha256`).
- **SHA-256 KHÔNG chứng minh được:** Chất lượng ngữ nghĩa của nhãn hay tính hợp pháp/an toàn của nội dung ảnh (một ảnh gán sai lớp `class_swap`, lệch box `bbox_drift`, dính dữ liệu nhạy cảm/khuôn mặt người thật, hoặc xuất phát từ nguồn vi phạm bản quyền vẫn cho ra một mã SHA-256 hợp lệ nếu file không bị thay đổi byte).

## 5. Truy vết
- **Tiêu chí phân tầng (`tier_a` và `tier_b`):**
  - Lọc các ảnh thuộc job sử dụng phiên bản guideline cũ (`v1.2` trước khi cập nhật lên `v1.3` theo `changelog.md`) và có chứa nhãn của lớp bị đổi quy tắc (`cup`, `id=41`).
  - **`tier_a` (3 ảnh):** Các ảnh thỏa mãn điều kiện trên **và** thuộc đúng các `scene_id` có phản ánh