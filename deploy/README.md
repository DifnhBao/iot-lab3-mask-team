---
title: Lab3 Mask Detection
emoji: 📷
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Cloud

Đây là template triển khai, chưa có endpoint Cloud thật.

Sau khi đã có models/best.pt và tài khoản Docker Space phù hợp:

1. Đặt các file server/, models/best.pt, requirements-server.txt trong repo Space.
2. Copy deploy/Dockerfile thành Dockerfile ở gốc repo Space.
3. Dùng nội dung README này làm README gốc để cấu hình sdk docker/app_port7860.
4. Build, xem startup logs và kiểm tra /health trả ready=true.
5. Đổi cloud_url client thành wss://<space-host>/ws/predict bằng hostname app service thực tế.

Không upload datasets, ảnh cá nhân hay config chứa secrets. CPU server là baseline cho lab; dùng model và streaming thực để đánh giá tốc độ sau deploy.
