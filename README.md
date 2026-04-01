# ✍️ IELTS-Writing-Evals

Repository này phục vụ nghiên cứu bài toán **chấm điểm IELTS Writing Task 2** theo nhiều hướng:
- Baseline encoder models (RoBERTa/ModernBERT/DistilRoBERTa).
- Fine-tune LLM (Qwen, Mistral) cho multi-output scoring.
- Inference ở 2 mức: **score-only inference** và **full inference với retriever + tool-using feedback agent**.

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

### 2.5 Minh hoạ thiết kế hệ thống (theo notebook `test_7_inference_retriever_full_zero_mistral_tool_use_automatic_6.ipynb`)

Notebook này thể hiện rõ kiến trúc **tool-using automatic pipeline** và là ví dụ đầy đủ nhất để hiểu cách hệ thống hoạt động đầu-cuối:

1. **Scoring phase**
   - Dùng model đa đầu ra để dự đoán `TR/CC/LR/GRA` (+ overall).

2. **Retriever phase (hybrid)**
   - Embedding bằng `all-MiniLM-L6-v2`.
   - Candidate retrieval theo cosine similarity.
   - Rerank bằng `final_score = vector_sim - 0.7 * quality_dist`.
   - `quality_dist` kết hợp khoảng cách điểm dự đoán và các retrieval features (`essay_len`, `prompt_relevance`, `lexical_diversity`, `readability_score`).

3. **Agent routing phase**
   - Agent chọn tuần tự tool hợp lệ theo state:
     - `predict_scores`
     - `retrieve_similar_essays`
     - `generate_feedback`
     - `verify_feedback`
     - `revise_feedback`
   - Có fallback policy khi parse action lỗi hoặc chọn sai tool.

4. **Verification & revision loop**
   - Kiểm tra feedback theo tiêu chí “specific / evidence-based / criterion-faithful”.
   - Chỉ revise tiêu chí bị lỗi và giới hạn retry theo từng criterion.

5. **Final output phase**
   - Trả về: predicted scores, retrieved cases, trace các bước tool-call, verdict verification, và feedback cuối cùng theo từng criterion.

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

Thư mục này chứa các notebook triển khai **pipeline suy luận đầy đủ**: scoring + retrieval + (tuỳ biến) feedback generation/tool routing.

#### `full_inference/test_7_inference_retriever_full.ipynb`
- Phiên bản full pipeline cơ sở cho nhánh test 7.

#### `full_inference/test_7_inference_retriever_full_zero.ipynb`
- Biến thể zero-shot của pipeline full inference.

#### `full_inference/test_7_inference_retriever_full_zero_mistral.ipynb`
- Biến thể dùng Mistral để generate/explain trong full inference.

#### `full_inference/test_7_inference_retriever_full_zero_mistral_tool_use.ipynb`
- Thêm cơ chế tool-use cho bước feedback.

#### `full_inference/test_7_inference_retriever_full_zero_mistral_tool_use_automatic_1.ipynb` → `..._6.ipynb`
- Chuỗi notebook tối ưu dần cơ chế routing tự động.
- `...automatic_6.ipynb` là bản minh hoạ rõ nhất cho thiết kế agent state-machine + verify/revise loop.

**Khi nào dùng `full_inference/` thay vì `score_inference/`?**
- Dùng `score_inference/` khi chỉ cần kiểm tra chất lượng đầu ra chấm điểm từ checkpoint.
- Dùng `full_inference/` khi cần đánh giá pipeline gần thực tế triển khai hơn (có retrieval, grounding cases, feedback generation và quality control).

---

## 4) Cách chạy đề xuất

### Bước 1: Chuẩn bị môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jupyter pandas numpy scikit-learn matplotlib seaborn transformers datasets torch peft accelerate sentence-transformers
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
6. Notebook trong `full_inference/` (ưu tiên `...automatic_6.ipynb` nếu muốn xem thiết kế agent đầy đủ)

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
  - Cơ chế verify/revise giới hạn retry ra sao.

---

## 6) Ghi chú

- Dự án đang ở dạng notebook research workflow, chưa đóng gói thành module Python hoàn chỉnh.
- Khi chuyển sang production, nên tách phần dùng chung (data processing, metrics, inference utils, retriever utils, agent orchestration) thành `src/`.
