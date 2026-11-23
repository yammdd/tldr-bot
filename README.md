# TL;DR Bot

Bot Discord đa server giúp tóm tắt 5–1.500 tin nhắn bị bỏ lỡ theo định dạng chuẩn **Meta AI (Messenger)** — phân biệt rõ **ai đã tham gia, ai nói gì, chủ đề nào đã chốt hoặc còn bỏ ngỏ**, kèm link nhảy tới tin nhắn gốc và mục **Nhắc đến bạn**.

<p align="center">
  <img src="assets/bot-avatar.png" alt="TL;DR Bot mark" width="128" height="128" />
</p>

---

## Kiến trúc & Tính năng

- **Tóm tắt hội thoại đa người nói (5 tầng xử lý):**
  - Giải mã `@mention`, `#channel`, phân biệt nickname trùng (`Tên (@username)`), giữ chuỗi trả lời `->(Tên #id)` để gỡ rối các cuộc nói chuyện xen kẽ.
  - Gộp tin nhắn liên tiếp cùng 1 người trong 2 phút, giữ lại các câu xác nhận ngắn (`ok`, `chốt`, `+1`), cắt gọn tin dán log dài quá 800 ký tự, chèn mốc ngắt phiên (> 30 phút).
  - Đếm token bằng `tiktoken` (tính toán cho tiếng Việt & đa ngôn ngữ): tự động chạy **Single-Pass (≤ 28K tokens)** hoặc **Session-Aware Map-Reduce song song** với trần an toàn **1.500 tin / 100K input tokens**.
  - Hậu kiểm kết quả JSON: xác thực mọi `ref_id` trước khi tạo Jump Link, chặn ping `@everyone`/`@here`, tách riêng mục **Nhắc đến bạn** (chỉ hiển thị riêng tư, tự ẩn khi chia sẻ ra kênh).
- **Bảo mật API Key & Phân quyền cho môi trường nhiều server:**
  - Nhập API Key & Endpoint qua `/config` bằng **Ephemeral Select + Modal ẩn** (không bao giờ xuất hiện trong lịch sử chat).
  - **Live Smoke Test bắt buộc:** Chạy kiểm tra completion thật cực nhẹ trước khi lưu; nếu lỗi sẽ báo chi tiết nguyên nhân và giữ nguyên cấu hình cũ.
  - **Write-Only AES-256-GCM trong SQLite WAL:** Mỗi server một bản ghi mã hóa riêng với nonce ngẫu nhiên 12-byte và AAD gắn cứng `guild_id` (chống hoán đổi ciphertext giữa các server). Hỗ trợ chuyển sang chế độ RAM-only qua `KEY_STORAGE_MODE=memory`.
  - **Phân quyền theo thứ bậc Role:** Chỉ **Chủ server (Owner)** hoặc thành viên có **Role cao hơn Role cao nhất của Bot** mới được dùng `/config`, `/reset_config` và `/language`.
  - **Chống SSRF & Ẩn bí mật:** Bắt buộc `https://`, chặn dải IP nội bộ/localhost, tự lọc ẩn mọi chuỗi key khỏi log và thông báo lỗi.
- **Sẵn sàng vận hành đa server (Multi-Server Scale):**
  - `AutoShardedBot` tự động sharding khi mở rộng sang nhiều server.
  - `ConcurrencyGuard` giới hạn số lượt chạy đồng thời theo từng server và toàn cục để một server bận không làm chậm các server khác.
  - Endpoint `GET /healthz` tích hợp sẵn cho Docker, Railway, Render, Fly.io hoặc Kubernetes.
- **Hỗ trợ mọi Endpoint OpenAI-compatible:** Có sẵn cấu hình cho **Google Gemini, DeepSeek, OpenAI, Anthropic Claude, OpenRouter, Groq** và **Custom Endpoint**.

---

## Cài đặt & Chạy Local

### 1. Thiết lập ứng dụng trên Discord Developer Portal
1. Tạo ứng dụng tại [Discord Developer Portal](https://discord.com/developers/applications) → mục **Bot**.
2. Tải lên avatar tối giản tại `assets/bot-avatar.png` (có sẵn file nguồn vector tại `assets/bot-avatar.svg`).
3. Bật **Privileged Gateway Intents → `MESSAGE CONTENT INTENT`** *(bắt buộc để đọc lịch sử tin nhắn)*.
4. Vào **OAuth2 → URL Generator**:
   - Scopes: `bot`, `applications.commands`
   - Permissions: `View Channels`, `Send Messages`, `Read Message History`, `Embed Links`
5. Mời bot vào server, sau đó vào **Server Settings → Roles** kéo Role của Bot nằm dưới Role Admin/Mod và cao hơn thành viên thường.

### 2. Chạy trên máy

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
Copy-Item .env.example .env
```

Điền `DISCORD_TOKEN` (và tạo `VAULT_MASTER_KEY` ngẫu nhiên) vào `.env`:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Khởi chạy bot:

```powershell
.\.venv\Scripts\python.exe run.py
```

---

## Deploy Multi-Server với Docker

```bash
cp .env.example .env
# Điền DISCORD_TOKEN và VAULT_MASTER_KEY vào .env
docker compose up -d --build
```

Kiểm tra trạng thái sức khỏe bot:

```bash
curl http://127.0.0.1:8080/healthz
```

*Lưu ý khi deploy lên VPS / Railway / Render / Fly.io:*
- Luôn gắn persistent volume cho `/app/data` và đặt cố định biến môi trường `VAULT_MASTER_KEY` (nếu đổi `VAULT_MASTER_KEY`, bot sẽ không giải mã được database cũ).
- Đặt `HEALTH_PORT=8080` để nền tảng cloud kiểm tra liveness qua `/healthz`.

---

## Danh sách lệnh

| Lệnh / Thao tác | Quyền hạn | Chức năng |
| :--- | :--- | :--- |
| `/config` | Owner hoặc Role > Bot | Chọn Provider / Custom Endpoint, nhập API Key qua Modal bảo mật + Live Smoke Test |
| `/model [model_id]` | Owner hoặc Role > Bot | Tự lấy danh sách model chat từ API hiện tại để chọn nhanh (không cần nhập lại API Key) |
| `/language` | Owner hoặc Role > Bot | Đổi ngôn ngữ mặc định của server (`vi`, `en`, `ja`, `ko`, `zh`, `auto`) |
| `/status` | Mọi thành viên | Xem trạng thái Provider, Model, ngôn ngữ & giới hạn ngữ cảnh (không hiển thị API Key) |
| `/reset_config` | Owner hoặc Role > Bot | Xóa API Key và cấu hình của server |
| `/help` | Mọi thành viên | Hướng dẫn nhanh cách thiết lập & sử dụng bot |
| `/tldr [count] [since] [style] [focus] [language]` | Mọi thành viên | Tóm tắt theo số tin (vd `500`), thời gian (`2h`, `24h`, `today`), phong cách & bộ lọc chủ đề |
| Chuột phải tin nhắn → **Apps → TL;DR từ tin nhắn này** | Mọi thành viên | Tóm tắt toàn bộ tin nhắn từ mốc đã chọn cho tới hiện tại |

---

## Kiểm thử tự động

```powershell
.\.venv\Scripts\pytest.exe -v
```
