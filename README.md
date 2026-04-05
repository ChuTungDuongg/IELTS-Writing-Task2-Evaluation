# ✍️ IELTS Writing Task 2 Evaluation

README này chỉ tập trung vào **2 notebook cốt lõi của hệ thống** + tóm tắt nhanh các notebook tiền xử lý theo yêu cầu.

---

## 🚀 1) Notebook train chính: `score_training/qwen_3b_test_8.ipynb`

### 🎯 Mục tiêu
Huấn luyện mô hình chấm điểm 4 tiêu chí IELTS Writing Task 2:
- **TR** (Task Response)
- **CC** (Coherence & Cohesion)
- **LR** (Lexical Resource)
- **GRA** (Grammatical Range & Accuracy)

### 🧠 Kiến trúc & dữ liệu
- Backbone: **Qwen/Qwen2.5-3B-Instruct** + **LoRA**.
- Input là prompt + essay (định dạng examiner prompt).
- Dùng thêm **nhóm feature thủ công theo từng tiêu chí** (TR/CC/LR/GRA) rồi fusion với hidden representation.
- Train/val/test lấy từ các file augmented:
  - `ielts_train_aug_df.csv`
  - `ielts_evals_aug_df.csv`
  - `ielts_test_locked_df.csv`

### ⚙️ Cấu hình đáng chú ý
- `max_length=1536`, `batch_size=6`, `grad_accum=4`
- `lr=4e-5`, `epochs=10`, `weight_decay=0.01`
- Chọn best model theo **`eval_mean_qwk`**
- Có `compute_metrics` cho MAE, QWK, within-0.5 accuracy

### 📈 Performance (đọc từ notebook)
- **Validation (best):**
  - `eval_mean_qwk = 0.6646`
  - `eval_mean_mae = 0.7535`
  - `eval_within_0.5_acc = 0.5767`
- **Test:**
  - `test_mean_qwk = 0.5747`
  - `test_mean_mae = 0.7583`
  - `test_within_0.5_acc = 0.5751`

### 📦 Output model
Notebook export dạng light package (LoRA adapter + custom heads + tokenizer + metadata), phục vụ inference.

---

## 🧩 2) Notebook full inference chính: `full_inference/test_8_inference_retriever_full_zero_mistral_tool_use_automatic_2.ipynb`

Đây là pipeline inference đầy đủ (không chỉ chấm điểm), gồm:

1. **🔢 Predict scores**
   - Load model scoring đã train (Qwen + LoRA + custom heads).
   - Dự đoán TR/CC/LR/GRA + Overall.

2. **🔍 Retrieval grounding**
   - Xây retrieval DB từ essay train.
   - Embedding bằng **`all-MiniLM-L6-v2`**.
   - Lấy các bài tương tự để làm ngữ cảnh khi sinh feedback.

3. **🤖 Tool-using feedback agent**
   - Dùng **Mistral-7B-Instruct-v0.3** cho explain/feedback theo tool workflow.
   - Bộ tool chính:
     - `predict_scores`
     - `detect_task_mismatch`
     - `retrieve_similar_essays`
     - `generate_feedback`
     - `verify_feedback`
     - `revise_feedback`

4. **✅ Verify & revise loop**
   - Kiểm tra feedback có bám sát bài viết, đúng tiêu chí, có evidence.
   - Nếu lỗi thì chỉ sửa phần tiêu chí lỗi, lặp theo giới hạn retry.

5. **🖥️ Demo UI**
   - Có cell Gradio để chạy end-to-end và hiển thị trace tool calls.

👉 Tóm lại: notebook này là **full hệ thống inference** (scoring + retriever + agent + quality control).

---

## 🛠️ 3) Tóm tắt nhanh các notebook hỗ trợ

### `feature_engineering.ipynb` 🧪
- Làm sạch dữ liệu, parse lại điểm TR/CC/LR/GRA từ `evaluation`.
- Tạo feature ngôn ngữ và feature semantic (prompt-essay relevance bằng SBERT).
- Tạo text instruction phục vụ train.
- Chia lại tập train/val/test ổn định để dùng xuyên suốt.

### `data_aug.ipynb` 🔁
- Kết hợp dữ liệu gốc HF với Kaggle Task 2.
- Chuẩn hóa cột, map score theo tiêu chí, loại trùng/lệch format.
- Tạo và lưu các file augmented chính:
  - `ielts_train_aug_df.csv`
  - `ielts_evals_aug_df.csv`
  - `ielts_test_locked_df.csv`

### `eda.ipynb` 📊
- Khám phá dữ liệu tổng: missing, duplicate, phân phối band.
- Phân tích độ dài prompt/essay/evaluation.
- Thống kê từ vựng phổ biến để hiểu đặc trưng corpus.

---

## 📌 4) Luồng chạy khuyến nghị (ngắn gọn)

1. `eda.ipynb`
2. `feature_engineering.ipynb`
3. `data_aug.ipynb`
4. `score_training/qwen_3b_test_8.ipynb`
5. `full_inference/test_8_inference_retriever_full_zero_mistral_tool_use_automatic_2.ipynb`

---

## 📝 Ghi chú
- Các notebook khác trong repo có thể là biến thể thử nghiệm; theo yêu cầu hiện tại có thể bỏ qua.
- Nếu cần demo nhanh hệ thống thật, ưu tiên notebook **full inference test_8**.
