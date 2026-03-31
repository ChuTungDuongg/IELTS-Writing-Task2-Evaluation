# ✍️ IELTS-Writing-Evals

Repository này phục vụ nghiên cứu bài toán **chấm điểm IELTS Writing Task 2** theo nhiều hướng:
- Baseline encoder models (RoBERTa/ModernBERT/DistilRoBERTa).
- Fine-tune LLM (Qwen, Mistral) cho multi-output scoring.
- Inference ở 2 mức: **score-only inference** và **full inference với retriever + generation**.

---

## 1) Mục tiêu dự án

- Phân tích dữ liệu essay IELTS và phân phối điểm theo từng tiêu chí.
- Xây dựng các pipeline huấn luyện để dự đoán điểm các tiêu chí Writing.
- So sánh nhiều kiến trúc/mức tài nguyên khác nhau (baseline nhỏ → mô hình lớn).
- Giữ toàn bộ workflow dưới dạng notebook để dễ tái lập và kiểm thử ý tưởng.

---

## 2) Tóm tắt cấu trúc hệ thống

> Luồng tổng quát: **Data → EDA/Feature Engineering → Training → Inference**.

### 2.1 Data layer (root CSV)
- `ielts_train_df.csv`: dữ liệu train.
- `ielts_val_df.csv`: dữ liệu validation.
- `ielts_test_df.csv`: dữ liệu test.

### 2.2 Analysis & Feature layer
- `eda.ipynb`: phân tích khám phá dữ liệu (phân phối điểm, độ dài bài viết, thiếu dữ liệu...).
- `feature engineering.ipynb`: tạo/đánh giá feature thủ công bổ sung cho mô hình.

### 2.3 Model training layer
- `baseline/`: các encoder baseline để lấy mốc (DistilRoBERTa, RoBERTa Base/Large, ModernBERT Large).
- `score_training/`: các notebook fine-tune LLM (Qwen 3B, Mistral 7B) cho bài toán scoring.

### 2.4 Inference layer
- `score_inference/`: suy luận điểm từ checkpoint theo từng test cấu hình.
- `full_inference/`: pipeline suy luận đầy đủ (retrieval + prompt/generation + scoring output), phù hợp khi cần mô phỏng pipeline gần thực tế hơn so với score-only.

---

## 3) Cấu trúc thư mục & vai trò từng file

### 3.1 Thư mục gốc

#### `README.md`
- Tài liệu mô tả tổng quan dự án, luồng hệ thống và vai trò từng notebook.

#### `reference.pdf`
- Tài liệu tham chiếu phục vụ đối chiếu trong quá trình nghiên cứu.

#### `eda.ipynb`
- Notebook EDA: kiểm tra kích thước dữ liệu, phân phối band điểm, bất thường dữ liệu, thống kê văn bản.

#### `feature engineering.ipynb`
- Notebook xây dựng và thử nghiệm đặc trưng thủ công cho essay.

#### `ielts_train_df.csv` / `ielts_val_df.csv` / `ielts_test_df.csv`
- Ba tập dữ liệu train/val/test dùng xuyên suốt các notebook.

---

### 3.2 `baseline/` – Baseline encoder

#### `baseline/distil_roberta_base_score.ipynb`
- Baseline nhẹ để kiểm tra pipeline nhanh và lấy mốc ban đầu.

#### `baseline/roberta_base_score.ipynb`
- Baseline cân bằng giữa chất lượng và chi phí.

#### `baseline/roberta_large_score.ipynb`
- Baseline backbone lớn để đo lợi ích khi scale model.

#### `baseline/modern_bert_large_score.ipynb`
- So sánh ModernBERT Large với họ RoBERTa.

---

### 3.3 `score_training/` – Fine-tune LLM

#### `score_training/Mistral_7B_ordinal_regress_1.ipynb`
- Thử nghiệm Mistral 7B theo hướng ordinal regression.

#### `score_training/qwen_3b_test_1.ipynb`
#### `score_training/qwen_3b_test_2.ipynb`
#### `score_training/qwen_3b_test_3.ipynb`
#### `score_training/qwen_3b_test_4.ipynb`
#### `score_training/qwen_3b_test_5.ipynb`
#### `score_training/qwen_3b_test_7.ipynb`
#### `score_training/qwen_3b_test_8.ipynb`
- Các biến thể cấu hình train Qwen 3B cho mục tiêu scoring theo tiêu chí.
- Mỗi file đại diện một thực nghiệm (khác nhau về config/loss/prompting/feature setup).

---

### 3.4 `score_inference/` – Score-only inference

#### `score_inference/test_1_inference.ipynb`
#### `score_inference/test_2_inference.ipynb`
#### `score_inference/test_7_inference.ipynb`
- Notebook chạy suy luận từ checkpoint tương ứng từng nhánh test.
- Mục tiêu chính: so sánh output dự đoán điểm giữa các cấu hình model đã train.

---

### 3.5 `full_inference/` – Full inference (phần quan trọng)

Thư mục này chứa các notebook triển khai **pipeline suy luận đầy đủ**, không chỉ dừng ở gọi model chấm điểm, mà còn gắn thêm thành phần retrieval/context để hỗ trợ đầu ra.

#### `full_inference/test_7_inference_retriever_full.ipynb`
- Phiên bản full pipeline cho nhánh test 7.
- Thường dùng làm baseline chính trong nhóm full inference có retriever.

#### `full_inference/test_7_inference_retriever_full_zero_1.ipynb`
- Biến thể “zero_1” của full pipeline.
- Dùng cho ablation/so sánh ảnh hưởng khi thay đổi cấu hình retrieval hoặc prompt ở chế độ zero-shot/near-zero.

#### `full_inference/test_7_inference_retriever_full_zero_mistral.ipynb`
- Biến thể full inference chuyển sang backbone/thiết lập Mistral.
- Phù hợp để so sánh cùng một pipeline retrieval nhưng khác model family.

**Khi nào dùng `full_inference/` thay vì `score_inference/`?**
- Dùng `score_inference/` khi chỉ cần kiểm tra chất lượng đầu ra chấm điểm từ checkpoint.
- Dùng `full_inference/` khi cần đánh giá pipeline gần thực tế triển khai hơn (có thêm retrieval/context orchestration).

---

## 4) Cách chạy đề xuất

### Bước 1: Chuẩn bị môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jupyter pandas numpy scikit-learn matplotlib seaborn transformers datasets torch peft accelerate
```

### Bước 2: Mở notebook

```bash
jupyter notebook
```

### Bước 3: Thứ tự chạy khuyến nghị

1. `eda.ipynb`
2. `feature engineering.ipynb`
3. Notebook trong `baseline/`
4. Notebook trong `score_training/`
5. Notebook trong `score_inference/`
6. Notebook trong `full_inference/` (khi cần test pipeline đầy đủ)

---

## 5) Gợi ý quản lý thực nghiệm

- Chuẩn hóa cách đặt tên run/checkpoint (ví dụ: `model_dataset_loss_seed_date`).
- Ghi lại metric quan trọng sau mỗi thí nghiệm (MAE/RMSE/QWK nếu có).
- Tách rõ train/val/test và cố định random seed để dễ tái lập.
- Với các notebook trong `full_inference/`, nên ghi rõ:
  - Retriever đang dùng gì.
  - Dữ liệu/context nguồn lấy từ đâu.
  - Prompt template phiên bản nào.
  - Mapping từ output text → score cuối cùng.

---

## 6) Ghi chú

- Dự án đang ở dạng notebook research workflow, chưa đóng gói thành module Python hoàn chỉnh.
- Khi chuyển sang production, nên tách phần dùng chung (data processing, metrics, inference utils, retriever utils) thành `src/`.
