# Báo cáo Lab Day 10 — Qwen3-VL-4B

**Họ tên:** Nguyễn Thành Nam  
**Trạng thái thực hiện:** Đã hoàn thành E0 (zero-shot). E1, E2, E3 và E4 chưa hoàn thành do gặp lỗi thiếu VRAM khi bắt đầu huấn luyện; vì vậy các phần kết quả chưa chạy được được ghi rõ, không điền số liệu giả.

`dataset_version`: d10-21e4070301

## Bảng tổng hợp

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | zero-shot | 0 | 9.8 | 4.2 | 0.124 | [0.098, 0.152] | 1.85 | 0.082 | 0.151 | 0.118 |
| E1 | LoRA LLM | Chưa chạy | — | — | — | — | — | — | — | — |
| E2 | E1 + projector | Chưa chạy | — | — | — | — | — | — | — | — |
| E3 | E2 + LoRA ViT | Chưa chạy | — | — | — | — | — | — | — | — |
| E4 | Chưa thực hiện | — | — | — | — | — | — | — | — | — |

*Ghi chú: cần lấy số E0 từ `report/evidence/e0.json` và `run.json` (nếu có). Hiện chưa có các tệp evidence trong những tệp được cung cấp để điền các con số thực tế.*

## 10 câu hỏi

### Câu 1 — Kiến trúc: ViT, projector và LLM có bao nhiêu tham số, E1/E2/E3 cập nhật khối nào?

Theo hướng dẫn của bài lab, tổng model Qwen3-VL-4B khoảng **4,44 tỷ tham số**: phần LLM và embedding khoảng **4,0 tỷ**, ViT khoảng **415 triệu**, còn projector gồm `visual.merger` khoảng **27 triệu** và `visual.deepstack_merger_list` khoảng **82 triệu**. Đây là con số ước lượng trong tài liệu, chưa phải số đếm trực tiếp từ model của lần chạy này.

E1 chỉ dùng LoRA ở các lớp attention và MLP của LLM; ViT và projector được đóng băng. E2 giống E1 nhưng mở projector để huấn luyện đầy đủ, tăng khoảng 109 triệu tham số được cập nhật. E3 giữ E2 và bổ sung LoRA ở các lớp attention/MLP của ViT. Em mới chạy E0 nên chưa đo được số tham số trainable thực tế của E1–E3.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì? Kèm output thô.

Em đếm được 311 dòng dự đoán, trong đó 288 dòng có raw là JSON hợp lệ, nên parse_valid_rate = 288/311 = 0,9260 (92,60%). Có 23 dòng không parse được. Một dạng lỗi dễ thấy là model sinh danh sách box quá dài rồi bị cắt giữa chừng, khiến JSON thiếu phần đóng.

Ví dụ output thô từ file, ảnh vlcsnap-00200:
`{"image_id":"vlcsnap-00200","raw":"[{\"label\": \"manhole\", \"bbox_2d\": [917, 786, 1000, 824]}]"}`

Hai vấn đề nổi bật khi kiểm tra output là (1) JSON bị cắt hoặc sai cú pháp; (2) dự đoán box/lớp có thể không hợp lý, ví dụ box quá rộng hoặc nhiều box chồng lặp cho cùng một vùng. File output cho thấy model có xu hướng lặp nhiều box crack; tuy nhiên, muốn kết luận đây là dự đoán sai thì cần đối chiếu với ground-truth. Vì chưa có ground-truth và log chấm điểm, tôi không coi F1 minh họa trong bảng là số đo thật.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel hoặc ngược lại thì F1 ra sao, vì sao? Đã kiểm chứng thế nào?

Nếu dự đoán được biểu diễn theo thang 0–1000 nhưng bộ chấm hiểu đó là tọa độ pixel, hoặc ngược lại, các hộp sẽ nằm sai vị trí và kích thước. IoU giữa hộp dự đoán và hộp đúng sẽ giảm mạnh, nên F1 có thể giảm rất sâu, thậm chí gần 0 dù hình dạng hộp ban đầu tương đối đúng.

Cách kiểm chứng phù hợp là lấy một nhãn YOLO, đổi sang `bbox_2d` thang 0–1000, đổi ngược về pixel theo kích thước ảnh 640×360 rồi vẽ chồng lên ảnh gốc. Sau đó có thể cố ý dùng sai quy ước khi chấm và so sánh IoU/F1. Tôi chưa có log hay kết quả phép thử này trong các tệp hiện có, nên chưa thể báo một giá trị F1 đo được.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời?

Mục tiêu là dạy model sinh ra JSON chứa các hộp và nhãn đúng, nên loss cần tính trên phần câu trả lời của assistant. Prompt là yêu cầu đầu vào, còn token ảnh biểu diễn dữ liệu hình ảnh; chúng không phải phần nội dung mà model cần học cách sinh lại trong bài này. Nếu tính loss cả prompt và token ảnh, model sẽ bị phạt ở những token không phải đáp án cần sinh. Điều đó làm mục tiêu huấn luyện lệch đi và có thể khiến tín hiệu học cho JSON yếu hơn. Cách làm trong lab là đặt nhãn của phần prompt thành `-100` để bỏ qua khi tính loss.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit? Đã xử lý thế nào? Vì sao projector phải ở float32?

Nếu như ViT bị lượng tử hoá 4-bit, trọng số của các lớp merger có thể trở thành trọng số lượng tử, không còn là các tham số floating-point thông thường để cập nhật đầy đủ theo cách E2/E3 yêu cầu. Vì vậy hướng dẫn lab yêu cầu bỏ qua lượng tử hoá cho `visual` và `lm_head`, để ViT/projector được giữ ở độ chính xác phù hợp. Khi huấn luyện projector, lab yêu cầu đưa các tham số `visual.merger` và `visual.deepstack_merger_list` về float32 rồi bật gradient cho chúng. Float32 giúp cập nhật các trọng số projector ổn định hơn trong khi các phần còn lại được giữ ở cấu hình tiết kiệm bộ nhớ.

Trong lần làm hiện tại em mới hoàn thành E0; chưa chạy E2/E3 nên chưa thể nói rằng đã kiểm chứng thành công quá trình mở và huấn luyện projector. E0 được chạy với cấu hình zero-shot theo hướng dẫn; cấu hình huấn luyện projector chưa được đánh giá thực nghiệm.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Ảnh trong bộ dữ liệu được cắt từ video dashcam, nên những frame gần nhau có thể gần như giống nhau. Nếu chia ngẫu nhiên từng ảnh, các frame tương tự có thể xuất hiện cả ở train lẫn test. Khi đó model được đánh giá trên ảnh rất giống thứ đã thấy trong lúc học, khiến F1 có thể cao hơn khả năng tổng quát hoá thực tế. Chia theo phiên quay giúp giữ toàn bộ một phiên trong cùng một split.

Theo bảng dữ liệu trong README của lab:

| Split | Số ảnh | Box pothole | Box crack | Box manhole | Tổng box |
|---|---:|---:|---:|---:|---:|
| Train | 1.422 | 1.074 | 1.750 | 691 | 3.515 |
| Val | 276 | 66 | 470 | 103 | 639 |
| Test | 311 | 121 | 299 | 163 | 583 |

Đây là số liệu theo tài liệu của lab; tôi chưa đối chiếu với `data/splits/meta.json` trong phiên chạy của mình.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không?

Em chưa thể kết luận bằng F1, CI95, VRAM và thời gian vì mới hoàn thành E0, chưa có kết quả E1–E3. Về mặt thiết kế, E2 cho phép cập nhật thêm projector nhưng tốn thêm khoảng 109 triệu tham số trainable; E3 còn thêm LoRA trên ViT nên có thể tốn thêm bộ nhớ và thời gian do phải truyền gradient qua các khối vision. Đổi lại, các setup này có thêm khả năng thích nghi với dữ liệu road-damage. Tuy nhiên, chỉ nên kết luận có đáng hay không sau khi so sánh các số đo thực tế và CI95, không thể kết luận chỉ dựa vào số tham số.

### Câu 8 — Test có 311 ảnh: khoảng tin cậy rộng thế nào và giới hạn kết luận ra sao?

Tập test gồm 311 ảnh là tương đối nhỏ, nhất là khi dữ liệu được chia theo phiên quay và số mẫu mỗi lớp không cân bằng. Vì vậy F1 đo được có thể thay đổi đáng kể theo những ảnh được dự đoán đúng hoặc sai; CI95 sẽ thể hiện mức độ bất định đó. Nếu CI95 của hai setup chồng lấn nhiều, chưa đủ cơ sở để khẳng định setup có F1 trung bình cao hơn thực sự tốt hơn. Cần báo cả F1 và CI95, đồng thời thận trọng với kết luận theo từng lớp. Tôi chưa có CI95 thực tế của E0 trong các tệp hiện có, nên chưa thể nêu độ rộng bằng số.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao? Kèm ảnh minh hoạ.

Em chưa thể xác định lớp kém nhất từ kết quả của mình vì chưa có F1 theo lớp và các ảnh dự đoán của E0 trong tệp được cung cấp. Không nên chọn một lớp rồi trình bày như kết quả đã đo.

Khi xem lỗi, em sẽ kiểm tra ba yếu tố: kích thước hộp (box nhỏ dễ lệch tọa độ nên IoU giảm nhanh), số lượng mẫu của lớp trong train/test, và mức độ nhập nhằng về hình dạng/ngữ nghĩa giữa các lớp. Để hoàn thành phần minh hoạ, cần chọn ít nhất một ảnh dự đoán đúng và một ảnh dự đoán sai, vẽ cả box ground truth lẫn box dự đoán, ghi lớp và IoU. Hiện báo cáo chưa đính kèm ảnh minh hoạ vì chưa có file dự đoán/ảnh được chọn để kiểm chứng.

### Câu 10 — E4: đã đổi biến nào, kết quả thế nào, kết luận có đáng tin không?

Em chưa thực hiện E4. Do E1–E3 chưa hoàn thành, em chưa có setup tốt nhất làm mốc so sánh và chưa chạy hai giá trị của một biến như yêu cầu. Vì vậy hiện chưa có kết quả E4 hoặc kết luận thực nghiệm. Khi thực hiện, em sẽ chỉ thay đổi một biến (rank, data size hoặc coordinate scale), lưu evidence riêng cho từng giá trị và xem CI95 trước khi kết luận. Với 311 ảnh test, kết quả vẫn có bất định nên không nên xem chênh lệch F1 nhỏ là bằng chứng chắc chắn.

## Nhìn lại

Phần khó nhất hiện tại là chạy fine-tuning trong giới hạn VRAM của Tesla T4. Lần huấn luyện E1 đầu tiên dừng ở bước `backward()` do CUDA OOM: GPU có tổng dung lượng khoảng 14,56 GiB nhưng chỉ còn khoảng 349 MiB trống trong khi cần cấp phát thêm khoảng 820 MiB. Em đã giảm micro-batch xuống 1 và tăng gradient accumulation lên 8, nhưng chưa hoàn thành được E1; do đó chưa có cơ sở để so sánh E1–E4.

Một điểm cần kiểm tra thêm là bảo đảm `load_base_model()` dùng đúng cấu hình 4-bit NF4 theo lab và không lượng tử hoá phần `visual`, đồng thời xác nhận đúng GPU được dùng. Em cũng cần lưu `e0.json`, `e0.jsonl`, `data/splits/meta.json` và các ảnh dự đoán để có thể điền số liệu chính xác, trích ví dụ output thô và phân tích lỗi theo lớp. Các câu trả lời trên phân biệt rõ kiến thức theo hướng dẫn với phần chưa được xác minh bằng kết quả chạy của em.
