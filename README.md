# ✍️ IELTS Writing Evals

> 📚 Repository phục vụ **nghiên cứu và đánh giá điểm bài viết IELTS Writing** bằng các mô hình ngôn ngữ. Dự án tập trung theo workflow notebook: EDA → feature engineering → baseline training → further training → inference.

## 🎯 Mục tiêu

- 🔍 Khám phá dữ liệu chấm điểm IELTS Writing theo từng tiêu chí.
- 🧠 Xây dựng pipeline huấn luyện và đánh giá các mô hình NLP.
- ⚖️ So sánh nhiều mô hình baseline và các biến thể fine-tuning.
- 🚀 Thử nghiệm mở rộng với Qwen trong thư mục `further_test/`.

## 🗂️ Cấu trúc repository

### Notebook chính
- 📊 `eda.ipynb`: Phân tích khám phá dữ liệu (Exploratory Data Analysis).
- 🛠️ `feature engineering.ipynb`: Tạo và xử lý đặc trưng.

### Baseline (`baseline/`)
- `distil_roberta_base_score.ipynb`
- `roberta_base_score.ipynb`
- `roberta_large_score.ipynb`
- `modern_bert_large_score.ipynb`

### Inference (`inference/`)
- `Inference.ipynb`: Luồng suy luận tổng quát.
- `Inference_grammar_feature.ipynb`: Inference có kết hợp grammar features.
- `B1_inference.ipynb`, `B2_inference.ipynb`: Inference theo từng biến thể mô hình.

### Thử nghiệm nâng cao (`further_test/`)
- `qwen_3b_3epochs.ipynb`: Pipeline baseline Qwen2.5-3B dạng **multi-output regression** (4 tiêu chí), fine-tune bằng **LoRA + HuggingFace Trainer**, chưa dùng grammar features.
- `qwen_3b_10epochs.ipynb`: Bản mở rộng từ baseline với huấn luyện lâu hơn và custom **WeightedLossTrainer** để gán trọng số loss theo từng tiêu chí (vẫn là regression 4 đầu ra).
- `qwen_3b_10epochs_grammar.ipynb`: Pipeline **multi-task** kết hợp embedding từ Qwen + **grammar feature engineering** thủ công; dùng nhánh chính dự đoán 4 tiêu chí và thêm thành phần loss bias/grammar để regularize.
- `qwen_3b_10epochs_grammar_FIX_B1.ipynb`: Biến thể tối ưu cho B1, vẫn là **multi-task regression + grammar features**, cân bằng lại criterion weights và xuất model nhẹ phục vụ inference.
- `qwen_3b_10epochs_grammar_FIX_B2.ipynb`: Biến thể B2 theo cùng phương pháp với B1 (multi-task + grammar features), điều chỉnh hyperparameter (ví dụ grad accumulation) cho ổn định hơn ở band mục tiêu B2.
- `qwen_3b_10epochs_grammar_FIX_B3.ipynb`: Biến thể B3 bổ sung **sample re-weighting theo band** (band-value weights) ngoài criterion weights, để giảm lệch phân phối điểm trong train.
- `qwen_3b_10epochs_grammar_FIX_B4.ipynb`: Biến thể B4 chuyển sang **ordinal regression** cho mỗi tiêu chí (mã hóa ngưỡng band và dùng BCEWithLogitsLoss trên các threshold), vẫn kết hợp các linguistic features cho cả 4 tiêu chí TR/TA CC LR GRA.

### Dữ liệu & tài liệu
- 🗃️ `ielts_train_df.csv`
- 🗃️ `ielts_val_df.csv`
- 🗃️ `ielts_test_df.csv`
- 📄 `reference.pdf`

## ⚡ Hướng dẫn sử dụng nhanh

### 1) Chuẩn bị môi trường

Khuyến nghị Python **3.10+** và Jupyter Notebook/Lab.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jupyter pandas numpy scikit-learn matplotlib seaborn transformers datasets torch
```

> 💡 Mỗi notebook có thể cần thêm thư viện riêng. Nếu gặp lỗi import, cài thêm theo cell đầu notebook tương ứng.

### 2) Chạy notebook

```bash
jupyter notebook
```

Thứ tự khuyến nghị:
1. `eda.ipynb`
2. `feature engineering.ipynb`
3. Notebook trong `baseline/`
4. Notebook trong `further_test/` (nếu cần thử nghiệm thêm)
5. Notebook trong `inference/`

### 3) Dữ liệu

Repository đã chứa sẵn file train/val/test. Nếu thay dataset:
- giữ nguyên schema cột mà notebook đang dùng,
- cập nhật đường dẫn file trong các cell đọc dữ liệu.

## 🧭 Gợi ý workflow tái lập kết quả

1. Chạy EDA để kiểm tra phân phối điểm và outlier.
2. Chốt tập đặc trưng ở `feature engineering.ipynb`.
3. Huấn luyện các baseline để lấy mốc.
4. Chạy các notebook Qwen để so sánh cải thiện.
5. Chạy inference trên tập test/ngoài tập để kiểm tra độ ổn định.

## 🤝 Đóng góp

Hoan nghênh đóng góp dưới các dạng:
- ➕ Thêm baseline notebook mới.
- ⚙️ Tối ưu pipeline huấn luyện/inference.
- 📈 Bổ sung metric đánh giá.
- 📝 Cập nhật tài liệu, README, hướng dẫn tái lập.

## 🧷 Lưu ý

- ⚠️ Dự án thiên về notebook nên kết quả có thể phụ thuộc môi trường chạy.
- 📦 Nên thêm `requirements.txt` hoặc `environment.yml` để tăng khả năng reproducible.

---

Nếu bạn muốn, mình có thể tiếp tục hỗ trợ tạo sẵn:
- `requirements.txt` tối thiểu cho toàn bộ notebook,
- checklist tái lập kết quả,
- README song ngữ Việt/Anh.
