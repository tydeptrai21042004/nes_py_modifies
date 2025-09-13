# Semi-fake ?

* Nhiều pipeline tuyên bố học từ ảnh RGB nhưng lại đọc trạng thái nội bộ RAM của game để tính reward, điều kiện dừng, metric — ví dụ Super Mario Bros dùng các địa chỉ \$006D (page X trong màn), \$0086 (tọa độ X trên màn). Đây là tín hiệu đặc quyền không tồn tại ngoài giả lập, nên tuy vẫn chạy môi trường thật, kết quả không còn thuần từ ảnh .
* Trong chuẩn Atari/ALE, cần chọn một trong hai chế độ quan sát: rgb hoặc ram. Nếu tuyên bố “từ ảnh”, không được pha trộn với ram.

# Lưu ý 

* NES cuộn nền qua thanh ghi PPUSCROLL (\$2005) ở PPU: nhân vật chạy sang phải thì nền cuộn sang trái. Trước khi camera đạt ngưỡng để cuộn, nền có thể đứng yên nên đo dịch chuyển toàn cục sẽ xấp xỉ 0.
* Bản đồ nút tay cầm NES trả về theo thứ tự A, B, Select, Start, Up, Down, Left, Right; nếu vô tình giữ Start sẽ pause → khung hình đứng yên, các phép đo từ ảnh sẽ sai.

# what i do to make it not fake

1. Khóa chế độ vision-only

   * Thêm wrapper VisionOnlyNES chặn truy cập env.ram để tránh rò rỉ trạng thái đặc quyền, giữ ranh giới rgb khác ram như tinh thần của ALE.

2. Reward từ ảnh, không dùng RAM

   * Thêm PixelShiftReward:

     * Đo dịch chuyển nền theo trục X bằng normalized cross-correlation (NCC) giữa các frame đã thu nhỏ (độ lệch < 0 nghĩa là nền trôi trái).
     * Hiệu chỉnh dấu: vì nền cuộn trái khi đi phải, reward quy ước tiến phải là dương.
     * Khi nền chưa cuộn, dùng template matching NCC theo dõi dịch chuyển cục bộ của nhân vật (player Δx), rồi chọn nguồn tín hiệu mạnh hơn làm reward. NCC/template matching bền vững với thay đổi sáng tương đối.

3. Bộ test vật lý từ ảnh (không dùng RAM)

   * Progress > NOOP: đi Right(+B) phải cho tổng reward từ ảnh cao hơn đứng yên — hoạt động cả giai đoạn tiền scroll (dựa player) lẫn sau khi camera cuộn (dựa nền).
   * Quỹ đạo nhảy parabol: theo dõi y(t) của sprite từ ảnh, fit y = a t² + b t + c cho a > 0 và R² cao → gia tốc gần hằng theo trục dọc.
   * Ma sát và giảm tốc: thả nút sau khi chạy phải → độ dịch chuyển theo khung giảm dần.
   * Onset cuộn camera: ban đầu nguồn motion là player, sau đó chuyển dần sang background khi camera bắt đầu cuộn, đúng hành vi PPUSCROLL.


summary: không dùng ram chỉ dùng frame
