# ✍️ IELTS Writing Evals

> 📚 Repository phục vụ **nghiên cứu và đánh giá điểm bài viết IELTS Writing** bằng các mô hình ngôn ngữ, bao gồm notebook cho EDA, feature engineering, baseline model và inference.

## 🎯 Mục tiêu

- 🔍 Khám phá dữ liệu chấm điểm IELTS Writing.
- 🧠 Xây dựng pipeline huấn luyện và đánh giá các mô hình NLP.
- ⚖️ So sánh nhiều mô hình baseline và biến thể fine-tuning.
- 🚀 Thử nghiệm thêm với các mô hình Qwen trong thư mục `further_test/`.

## 🗂️ Cấu trúc thư mục

- 📊 `eda.ipynb`: Phân tích khám phá dữ liệu (Exploratory Data Analysis).
- 🛠️ `feature engineering.ipynb`: Xử lý và tạo đặc trưng cho dữ liệu.
- 🔮 `inference/Inference.ipynb`: Notebook chạy suy luận (inference).
- 🧪 `baseline/`: Các notebook baseline với nhiều kiến trúc khác nhau:
  - `distil_roberta_base_score.ipynb`
  - `roberta_base_score.ipynb`
  - `roberta_large_score.ipynb`
  - `modern_bert_large_score.ipynb`
- 🧬 `further_test/`: Thử nghiệm bổ sung với Qwen 3B:
  - `qwen_3b_3epochs.ipynb`
  - `qwen_3b_10epochs.ipynb`
  - `qwen_3b_13epochs.ipynb`
- 🗃️ Các file dữ liệu:
  - `ielts_train_df.csv`
  - `ielts_val_df.csv`
  - `ielts_test_df.csv`
- 📄 `reference.pdf`: Tài liệu tham khảo.

## ⚡ Hướng dẫn sử dụng nhanh

### 1️⃣ Chuẩn bị môi trường

✅ Khuyến nghị dùng Python 3.10+ và Jupyter Notebook/Lab.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jupyter pandas numpy scikit-learn matplotlib seaborn transformers datasets torch
```

> 💡 Tuỳ từng notebook, bạn có thể cần cài thêm thư viện khác.

### 2️⃣ Mở notebook

```bash
jupyter notebook
```

Sau đó chạy lần lượt các notebook theo mục tiêu:
- 🔍 Khám phá dữ liệu: `eda.ipynb`
- 🛠️ Tạo đặc trưng: `feature engineering.ipynb`
- 🧪 Huấn luyện/đánh giá baseline: thư mục `baseline/`
- 🔮 Suy luận: `inference/Inference.ipynb`

### 3️⃣ Dữ liệu

📌 Repository đã có sẵn các file CSV train/val/test. Nếu bạn thay đổi dữ liệu đầu vào, hãy cập nhật đường dẫn trong notebook tương ứng.

## 🧭 Gợi ý quy trình làm việc

1. 🔎 Chạy `eda.ipynb` để hiểu phân phối điểm và chất lượng dữ liệu.
2. 🧱 Chạy `feature engineering.ipynb` để chuẩn bị input.
3. 🧪 Chạy các notebook trong `baseline/` để lấy mốc so sánh.
4. 🚀 Chạy thử nghiệm nâng cao trong `further_test/` nếu cần.
5. 🔮 Dùng `inference/Inference.ipynb` để kiểm tra mô hình trên dữ liệu mới.

## 🤝 Đóng góp

Bạn có thể đóng góp bằng cách:
- ➕ Bổ sung notebook baseline mới.
- ⚙️ Tối ưu pipeline huấn luyện.
- 📈 Cải thiện phần đánh giá/chấm điểm.
- 📝 Cập nhật README và tài liệu sử dụng.

## 🧷 Lưu ý

- ⚠️ Đây là repository thiên về notebook, nên việc tái lập kết quả có thể phụ thuộc môi trường chạy.
- 📦 Nên cố định phiên bản thư viện (`requirements.txt`) nếu muốn reproducible tốt hơn.

---

🌟 Nếu bạn muốn, mình có thể tiếp tục tạo thêm:
- `requirements.txt` tối thiểu cho toàn bộ notebook,
- mẫu cấu trúc `src/` để chuyển dần từ notebook sang code production,
- hoặc README tiếng Anh song song với bản tiếng Việt này.
