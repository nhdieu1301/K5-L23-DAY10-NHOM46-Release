# Bảo mật khi làm việc

> Gắn với lô lab250_ff60d0167c45 (seed 46007) và lô bàn giao lab250_ff60d0167c45_v2.

## 1. Ca USB: hết giờ, muốn chép lô về nhà làm tiếp
Không chép lô ra USB hay máy cá nhân. Lô 250 ảnh này có ảnh người thật (COCO) và nhãn chưa phát hành (v1 đang HOLD), ra khỏi máy công ty là mất kiểm soát, không thu hồi được nếu USB thất lạc. Gửi qua email cá nhân cũng là đưa dữ liệu ra ngoài hệ thống, còn nhờ người khác làm hộ thì người đó không được giao job, không có trách nhiệm và để lại dấu vết sai người trong truyvet.csv. Cách đúng: dừng ở mốc hiện tại, ghi tiến độ vào review_log/ghi chú job, báo lead xin gia hạn hoặc xin truy cập từ xa (VPN, máy ảo công ty) nếu được duyệt.

## 2. Năm việc: được hay không
- Mở job được giao trên máy công ty: **được**, đúng người, đúng thiết bị, đúng quyền theo jobs.csv.
- Chụp màn hình gửi người quen: **không**, ảnh lô có mặt người thật và nhãn nội bộ, là rò rỉ dữ liệu dù chỉ một ảnh.
- Chép lô ra USB: **không**, dữ liệu rời khỏi môi trường được kiểm soát, không truy vết được (xem mục 1).
- Dán nhãn thật lên chatbot: **không**, nội dung bị gửi tới dịch vụ bên ngoài và có thể bị lưu lại; chỉ hỏi AI bằng ví dụ giả hoặc đoạn code không chứa dữ liệu lô.
- Đưa tài khoản CVAT cho đồng nghiệp: **không**, mọi thao tác sẽ ghi dưới tên mình, phá vỡ truy vết annotator/reviewer; đồng nghiệp cần được cấp quyền riêng.

## 3. Ca khuôn mặt: thấy mặt người rõ trong lô, guideline yêu cầu làm mờ
Kết luận: **Hold** lô (không phát hành ảnh đó ra cho team AI) và báo ngay người phụ trách dữ liệu/lead kèm tên ảnh. Không tự làm mờ hay xoá ảnh vì sẽ làm đổi sha256, lệch manifest và nhãn, phá v1 đã giao; việc làm mờ phải do người có trách nhiệm quyết định và tạo bản mới có lineage. Ví dụ trong lô của nhóm: ảnh 000000024243 (hai phụ nữ ngồi cạnh tủ lạnh trên vỉa hè, mặt nhìn rõ) và 000000021604 (chân dung người đàn ông đeo kính, cà vạt đèn LED).
