# Trọng số model

Đặt `best.pt` của lượt huấn luyện khẩu trang tại đây.

Class names phải đúng:

```text
0: with_mask
1: without_mask
2: mask_incorrect
```

Trọng số COCO `yolov8n.pt` chỉ dùng khởi tạo huấn luyện; server sẽ từ chối dùng nó làm model khẩu trang.

Chưa có `best.pt` trong dự án hiện tại. Server báo health503 và lỗi thiếu model cho tới khi có weights hợp lệ.
