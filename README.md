# ✍️ IELTS-Writing-Evals

Repository này phục vụ nghiên cứu bài toán **chấm điểm IELTS Writing Task 2** bằng nhiều hướng tiếp cận:
- Baseline encoder models (RoBERTa/ModernBERT/DistilRoBERTa).
- Fine-tune LLM (Qwen, Mistral) cho multi-output scoring.
- Inference notebook để thử nghiệm checkpoint đã train.

---

## 1) Mục tiêu dự án

- Phân tích dữ liệu essay IELTS và phân phối điểm theo từng tiêu chí.
- Xây dựng các pipeline huấn luyện để dự đoán điểm các tiêu chí Writing.
- So sánh nhiều kiến trúc/mức tài nguyên khác nhau (baseline nhỏ → mô hình lớn).
- Giữ toàn bộ workflow dạng notebook để dễ tái lập thí nghiệm.

---

## 2) Cấu trúc thư mục & giải thích chi tiết từng file

> Bạn có thể xem repo theo luồng: **dữ liệu → khám phá/feature → training → inference**.

### 2.1. Thư mục gốc

#### `README.md`
- Tài liệu mô tả tổng quan dự án, cấu trúc thư mục, vai trò từng notebook.

#### `reference.pdf`
- Tài liệu tham chiếu (đề bài/ghi chú/nguồn tham khảo nội bộ của dự án).
- Dùng làm ngữ cảnh khi đọc/đối chiếu kết quả trong notebook.

#### `eda.ipynb`
- Notebook phân tích khám phá dữ liệu (EDA).
- Thường bao gồm:
  - Kiểm tra kích thước tập dữ liệu.
  - Phân phối điểm theo tiêu chí.
  - Kiểm tra thiếu dữ liệu / bất thường.
  - Quan sát độ dài bài viết và các đặc tính bề mặt.

#### `feature engineering.ipynb`
- Notebook xây dựng đặc trưng thủ công cho essay.
- Mục đích:
  - Trích xuất feature ngôn ngữ học bổ sung cho mô hình.
  - So sánh hiệu quả giữa biểu diễn thuần transformer và hướng hybrid (embedding + handcrafted features).

#### `ielts_train_df.csv`
- Tập huấn luyện chính.
- Chứa essay/prompt/nhãn điểm dùng để train model.

#### `ielts_val_df.csv`
- Tập validation.
- Dùng chọn checkpoint, tinh chỉnh siêu tham số, theo dõi overfitting.

#### `ielts_test_df.csv`
- Tập test.
- Dùng đánh giá cuối cùng và kiểm tra khả năng tổng quát hóa.

---

### 2.2. `baseline/` – Các mô hình baseline

#### `baseline/distil_roberta_base_score.ipynb`
- Baseline nhẹ, huấn luyện nhanh.
- Phù hợp lấy mốc ban đầu, kiểm tra pipeline end-to-end trước khi chạy mô hình lớn.

#### `baseline/roberta_base_score.ipynb`
- Baseline chuẩn với RoBERTa base.
- Cân bằng tốt giữa chất lượng và chi phí huấn luyện.

#### `baseline/roberta_large_score.ipynb`
- Phiên bản RoBERTa large.
- Dùng để kiểm tra liệu tăng kích thước backbone có cải thiện điểm đáng kể hay không.

#### `baseline/modern_bert_large_score.ipynb`
- Baseline với ModernBERT large.
- Mục tiêu so sánh kiến trúc encoder mới hơn với họ RoBERTa truyền thống.

---

### 2.3. `score_training/` – Thí nghiệm fine-tune LLM

Nhóm notebook này tập trung vào train các mô hình lớn (Qwen/Mistral), chủ yếu theo hướng multi-criterion scoring.

#### `score_training/Mistral_7B_3epochs_ordinal_regress_1.ipynb`
- Thử nghiệm Mistral 7B theo hướng ordinal regression.
- Nhắm tới đặc tính thứ bậc của band điểm IELTS.

#### `score_training/qwen_3b_10epochs_test_1.ipynb`
- Thử nghiệm Qwen 3B (10 epochs), cấu hình test 1.
- Đóng vai trò mốc thực nghiệm đầu của nhánh Qwen 10 epochs.

#### `score_training/qwen_3b_10epochs_test_2.ipynb`
- Biến thể cấu hình test 2.
- So sánh thay đổi về loss/feature/training setup so với test 1.

#### `score_training/qwen_3b_10epochs_test_3.ipynb`
- Biến thể cấu hình test 3.
- Thường dùng để xác nhận độ ổn định khi thay đổi một phần kiến trúc hoặc hyperparameters.

#### `score_training/qwen_3b_10epochs_test_4.ipynb`
- Biến thể cấu hình test 4.
- Dùng cho ablation/so sánh incremental improvement.

#### `score_training/qwen_3b_10epochs_test_5.ipynb`
- Biến thể cấu hình test 5.
- Tập trung tinh chỉnh để cân bằng chất lượng dự đoán và độ ổn định train.

#### `score_training/qwen_3b_10epochs_test_7.ipynb`
- Biến thể cấu hình test 7.
- Mở rộng thêm hướng feature/prompting theo tiêu chí (tuỳ phiên bản notebook).

#### `score_training/qwen_3b_10epochs_test_8.ipynb`
- Biến thể cấu hình test 8.
- Thường là một trong các bản tối ưu throughput và/hoặc chất lượng trong nhánh Qwen.

> Gợi ý: vì các file đặt tên `test_X`, bạn nên ghi log thực nghiệm (metric + config chính) ở đầu/cuối mỗi notebook để tiện truy vết kết quả.

---

### 2.4. `score_inference/` – Notebook suy luận

#### `score_inference/test_1_inference.ipynb`
- Inference cho nhánh model tương ứng test 1.
- Dùng để kiểm tra output prediction trên dữ liệu mới hoặc tập test.

#### `score_inference/test_2_inference.ipynb`
- Inference cho biến thể test 2.
- Dùng so sánh trực tiếp chất lượng đầu ra giữa các checkpoint.

#### `score_inference/test_7_inference.ipynb`
- Inference cho biến thể test 7.
- Thường dùng đánh giá các thay đổi mạnh về feature/prompting sau khi train.

---

## 3) Cách chạy đề xuất

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

---

## 4) Gợi ý quản lý thực nghiệm

- Chuẩn hóa cách đặt tên run/checkpoint (ví dụ: `model_dataset_loss_seed_date`).
- Ghi lại metric quan trọng sau mỗi thí nghiệm (MAE/RMSE/QWK nếu có).
- Tách rõ dữ liệu train/val/test và cố định random seed để tái lập.
- Nếu thêm notebook mới, cập nhật lại mục “Cấu trúc thư mục” ngay để người khác theo dõi dễ hơn.

---

## 5) Ghi chú

- Dự án hiện thiên về notebook research workflow, chưa đóng gói thành module Python hoàn chỉnh.
- Khi chuyển sang production/integration, nên tách code chung (data processing, model class, metrics, inference utils) thành thư mục `src/` để dễ test và tái sử dụng.
