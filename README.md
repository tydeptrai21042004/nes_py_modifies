# Semi-fake ?

## 1) Họ lấy những thông tin gì từ môi trường?

Trình giả lập cung cấp hai luồng dữ liệu chính cho Python:

* Khung hình RGB: framebuffer 256×240 do PPU kết xuất (những gì xuất hiện trên màn hình). Phần này được lộ ra qua hàm C++ `Screen()` và dùng làm observation trong Gym (hộp `Box(240, 256, 3)`).
* RAM CPU (2 KiB): con trỏ trực tiếp tới RAM chính của NES trong dải địa chỉ \$0000–\$07FF, lộ ra qua hàm C++ `Memory()` và được bọc ở Python thành `env.ram` (một view NumPy). Thực hành phổ biến là đọc các byte cụ thể để khôi phục trạng thái game như world/stage của Mario, tọa độ x/y, đồng hồ thời gian, số mạng, số coin,… (các giá trị này nằm ở địa chỉ đã biết).

Trên nền tảng nes-py, gói `gym-super-mario-bros` sử dụng rõ ràng các biến lấy từ RAM để tính reward và trả về `info`:

* Reward dạng: `r = (x1 - x0) + (clock_delta) + (death_penalty)` nên tối thiểu cần vị trí X, đồng hồ, và trạng thái chết/sống.
* `info` chứa các khóa như `x_pos`, `y_pos`, `world`, `stage`, `time`, `status`, `coins`, `life`, `score`, `flag_get`,… — đều là các đại lượng đọc trực tiếp từ trạng thái nội bộ (RAM), không phải suy ra từ ảnh.


Với Super Mario Bros. cụ thể, cộng đồng đã ánh xạ chi tiết RAM, ví dụ:

* \$006D: “horizontal page” (phần bù trang theo trục X)
* \$0086: “on-screen X” (tọa độ X trên màn hình)
* \$009F: vận tốc thẳng đứng (phần nguyên; phần thập phân nằm ở địa chỉ khác)

Tọa độ X toàn cục thường ghép: `x = 256 * RAM[$006D] + RAM[$0086]`.

Summary: phần “họ lấy” gồm (a) khung hình RGB được kết xuất, và (b) các biến vô hướng “ground-truth” lấy từ RAM nội bộ (tọa độ X/Y của Mario, đồng hồ, mạng sống, v.v.), dùng cho shaping reward, điều khiển vòng đời episode, và báo cáo metric.

## 2) Thiếu gì hoặc bị mất thông tin nếu chỉ dựa vào dữ liệu đã lấy?

Tùy kênh dữ liệu đang dựa vào:

### Chỉ dựa vào vài byte RAM (như cách tính reward của wrapper)

* Không nắm trọn động lực học: thường chỉ lấy vị trí X và đồng hồ, cộng thêm cờ chết/sống; bỏ sót động lực học theo trục dọc (y, vy), cờ “đang chạm đất”/“đang trên không”, trạng thái va chạm, tương tác kẻ địch,… trừ khi chủ động đọc thêm địa chỉ tương ứng. N
* Dễ hiểu sai vị trí nếu không kết hợp đúng “page + on-screen X” với trạng thái camera/scroll. Trên NES, chuyển động màn chơi chủ yếu thể hiện qua cuộn nền của PPU (thanh ghi `$2005`/PPUSCROLL). Nếu chỉ dùng on-screen X mà không tính page/scroll, có thể lẫn lộn chuyển động camera với chuyển động nhân vật. Scroll do PPU xử lý, không nằm trong 2 KiB RAM CPU trả về bởi `Memory()`.

### Chỉ dựa vào ảnh

* Ảnh là kết quả raster cuối cùng: không thấy đối tượng ngoài khung hình, biến ẩn, hay trạng thái vật lý chính xác (tọa độ trong hệ toạ độ level, vận tốc chính xác, pháp tuyến va chạm). Nhiều trạng thái nằm trong các thanh ghi PPU và bộ nhớ sprite/OAM/nametable — không hiện trực tiếp trong 2 KiB RAM của CPU — và chỉ thể hiện gián tiếp trong khung hình.


### Chỉ dựa vào 2 KiB RAM CPU

* Không thấy trực tiếp các thanh ghi PPU (\$2000–\$2007) điều khiển scroll và kết xuất; 
### Một số thiếu hụt gặp trong `info`/reward mặc định

* Trạng thái camera/scroll: các biến/cờ quyết định khi nào camera bắt đầu cuộn, điểm mép màn hình, ngưỡng scroll,…  không có trong `info`.
* Chi tiết scroll tinh của PPU: coarse/fine X,Y do `$2005` quản lý không có trong `info`.
* Thành phần vận tốc vi phân/sub-pixel: `info` có `x_pos`/`y_pos` thiếu các phần phân số hoặc tách nguyên-phân số của vận tốc (ví dụ vận tốc dọc có byte riêng phần nguyên và địa chỉ khác cho phần phân số), hữu ích để nhận biết gia tốc, ma sát, hay quỹ đạo nhảy.
* Trạng thái thế giới khác: vị trí kẻ địch, ngưỡng scroll, cờ va chạm,… tồn tại trong RAM nhưng không được surface lên `info`, dù có thể giúp tạo metric giàu hơn hoặc điều kiện dừng an toàn hơn.
* Ngữ cảnh thời gian bước: `step` có thể frameskip; `info` mặc định không chú thích khung nào bị bỏ qua, gây khó khi suy luận vật lý theo thời gian nếu chỉ dựa vào observation.


#### Lưu ý 

* NES cuộn nền qua thanh ghi PPUSCROLL (\$2005) ở PPU: nhân vật chạy sang phải thì nền cuộn sang trái. Trước khi camera đạt ngưỡng để cuộn, nền đứng yên nên đo dịch chuyển toàn cục sẽ xấp xỉ 0.


# what i do to make it not fake

1. Khóa chế độ vision-only

   * Thêm wrapper VisionOnlyNES chặn truy cập env.ram để tránh rò rỉ trạng thái đặc quyền, giữ ranh giới rgb khác ram như tinh thần của ALE.

2. Reward từ ảnh

   * Thêm PixelShiftReward:

     * Đo dịch chuyển nền theo trục X bằng normalized cross-correlation (NCC) giữa các frame đã thu nhỏ (độ lệch < 0 nghĩa là nền trôi trái).
     * Hiệu chỉnh dấu: vì nền cuộn trái khi đi phải, reward quy ước tiến phải là dương.
     * Khi nền chưa cuộn, dùng template matching NCC theo dõi dịch chuyển cục bộ của nhân vật (player Δx), rồi chọn nguồn tín hiệu mạnh hơn làm reward. NCC/template matching bền vững với thay đổi sáng tương đối.

3. Bộ test vật lý từ ảnh 

   * Progress > NOOP: đi Right(+B) phải cho tổng reward từ ảnh cao hơn đứng yên — hoạt động cả giai đoạn tiền scroll (dựa player) lẫn sau khi camera cuộn (dựa nền).
   * Quỹ đạo nhảy gần parabol: theo dõi y(t) của sprite từ ảnh, fit y = a t² + b t + c cho a > 0 và R² cao → gia tốc gần hằng theo trục dọc.
   * Ma sát và giảm tốc: thả nút sau khi chạy phải → độ dịch chuyển theo khung giảm dần.
   * Onset cuộn camera: ban đầu nguồn motion là player, sau đó chuyển dần sang background khi camera bắt đầu cuộn, đúng hành vi PPUSCROLL.


summary: không dùng ram chỉ dùng frame




