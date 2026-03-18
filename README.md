# ✍️ IELTS Writing Evals

> 📚 Repository phục vụ **nghiên cứu và đánh giá điểm bài viết IELTS Writing** bằng các mô hình ngôn ngữ. Dự án đi theo workflow notebook: **EDA → feature engineering → baseline training → further experiments → inference**.

## 🎯 Mục tiêu

- 🔍 Khám phá dữ liệu chấm điểm IELTS Writing theo từng tiêu chí.
- 🧠 Xây dựng pipeline huấn luyện và đánh giá các mô hình NLP/LLM cho bài toán chấm điểm.
- ⚖️ So sánh nhiều mô hình baseline với các biến thể fine-tuning nâng cao.
- 🚀 Thử nghiệm thêm với họ mô hình **Qwen 2.5-3B** trong thư mục `further_test/`.

## 🗂️ Cấu trúc repository

### Notebook chính
- 📊 `eda.ipynb`: Phân tích khám phá dữ liệu (phân phối điểm, chất lượng dữ liệu, outlier, đặc điểm prompt/bài viết).
- 🛠️ `feature engineering.ipynb`: Tạo các đặc trưng thủ công phục vụ mô hình hoặc so sánh với embedding-based approach.

### Baseline (`baseline/`)
- `distil_roberta_base_score.ipynb`: Baseline gọn nhẹ để có mốc nhanh.
- `roberta_base_score.ipynb`: Baseline RoBERTa tiêu chuẩn cho bài toán scoring.
- `roberta_large_score.ipynb`: Bản RoBERTa lớn hơn để kiểm tra trade-off giữa chất lượng và chi phí.
- `modern_bert_large_score.ipynb`: Baseline với backbone ModernBERT Large.

### Inference (`inference/`)
- `Inference.ipynb`: Luồng suy luận tổng quát.
- `Inference_grammar_feature.ipynb`: Inference cho mô hình có dùng grammar features.
- `B1_inference.ipynb`, `B2_inference.ipynb`, `B7_inference.ipynb`: Inference cho các biến thể checkpoint cụ thể.

## 🔬 Giải thích chi tiết thư mục `further_test/`

Thư mục này chứa các notebook thử nghiệm nâng cao quanh backbone **`Qwen/Qwen2.5-3B-Instruct`**, chủ yếu dùng **LoRA/PEFT**, token hóa essay + prompt thành một input hợp nhất, rồi dự đoán **4 tiêu chí IELTS Writing**. Có thể hiểu các notebook như một chuỗi tiến hóa: từ baseline regression thuần văn bản, sang kết hợp grammar features, rồi tiến thêm tới reweighting, feature theo từng tiêu chí, và ordinal regression.

### 1) `qwen_3b_3epochs.ipynb`
**Vai trò:** baseline Qwen tối giản để lấy mốc ban đầu.

**Phương pháp / hướng tiếp cận:**
- Dùng `Qwen2.5-3B-Instruct` làm backbone.
- Chuyển bài toán thành **multi-output regression** với 4 đầu ra tương ứng 4 tiêu chí chấm điểm.
- Input được ghép theo dạng:
  - `[PROMPT] ...`
  - `[ESSAY] ...`
- Fine-tune bằng **LoRA** để giảm chi phí huấn luyện.
- Dùng `Trainer` chuẩn của Hugging Face, chưa thêm cơ chế weighting đặc biệt.
- Huấn luyện **3 epochs**, phù hợp để kiểm tra nhanh pipeline có chạy ổn định không.

**Khi nào nên dùng notebook này:**
- Khi muốn có baseline đơn giản, dễ tái lập.
- Khi cần xác minh dữ liệu, tokenizer, loss, metric đang hoạt động đúng trước khi thử nghiệm phức tạp hơn.

### 2) `qwen_3b_10epochs.ipynb`
**Vai trò:** mở rộng baseline bằng huấn luyện lâu hơn và tinh chỉnh loss.

**Phương pháp / hướng tiếp cận:**
- Vẫn là **multi-output regression** trên 4 tiêu chí.
- Tăng số vòng huấn luyện lên **10 epochs** để khai thác kỹ hơn backbone Qwen.
- Dùng `WeightedLossTrainer` để áp **loss weight theo từng tiêu chí**, thay vì coi mọi criterion quan trọng như nhau.
- Vẫn không dùng grammar features thủ công; mô hình chủ yếu học từ biểu diễn ngữ cảnh của prompt + essay.

**Ý tưởng cốt lõi:**
- Nếu một số tiêu chí khó học hơn hoặc quan trọng hơn, có thể tăng trọng số loss tương ứng.
- Đây là bước chuyển từ “baseline thuần” sang “baseline có điều khiển ưu tiên tối ưu”.

### 3) `qwen_3b_10epochs_grammar.ipynb`
**Vai trò:** thêm tri thức thủ công về ngữ pháp vào mô hình.

**Phương pháp / hướng tiếp cận:**
- Trích xuất **grammar features** từ essay như độ dài câu, mật độ từ, tín hiệu cấu trúc câu hoặc các chỉ báo gần với chất lượng ngữ pháp.
- Xây dựng mô hình **multi-task / hybrid**:
  - Một nhánh dùng embedding từ Qwen.
  - Một nhánh dùng `gra_features` thủ công.
- Trainer tùy chỉnh (`IELTSMultiTaskTrainer`) kết hợp nhiều thành phần loss.
- Có thêm thành phần **bias/auxiliary loss** để regularize quá trình học.

**Ý tưởng cốt lõi:**
- Điểm IELTS Writing không chỉ phụ thuộc vào ngữ nghĩa tổng thể, mà còn bị ảnh hưởng bởi tín hiệu ngôn ngữ học bề mặt.
- Kết hợp LLM embedding với feature engineering giúp mô hình “nhìn” rõ hơn các dấu hiệu grammar quality.

### 4) `qwen_3b_10epochs_grammar_FIX_B1.ipynb`
**Vai trò:** bản tinh chỉnh của notebook grammar, ưu tiên tính ổn định và khả năng suy luận cho biến thể B1.

**Phương pháp / hướng tiếp cận:**
- Kế thừa kiến trúc **Qwen + grammar features + multi-task regression**.
- Duy trì trainer tùy chỉnh và cơ chế phối hợp nhiều loss thành phần.
- Tối ưu lại hyperparameter/weight để checkpoint cho nhánh B1 chạy ổn định hơn.
- Hướng tới xuất bản model “gọn” hơn phục vụ inference sau huấn luyện.

**Cách hiểu notebook này:**
- Không phải đổi triết lý mô hình hoàn toàn.
- Chủ yếu là một bản “fix” mang tính thực nghiệm để ổn định kết quả cho một biến thể checkpoint cụ thể.

### 5) `qwen_3b_10epochs_grammar_FIX_B2.ipynb`
**Vai trò:** biến thể B2 của cùng họ mô hình grammar-aware.

**Phương pháp / hướng tiếp cận:**
- Giữ nguyên hướng **hybrid text + grammar features**.
- Tiếp tục dùng **multi-output regression** thay vì ordinal formulation.
- Điều chỉnh thông số huấn luyện như batch/gradient accumulation/cân bằng loss để phù hợp hơn với checkpoint B2.

**Ý nghĩa thực nghiệm:**
- Notebook này phục vụ so sánh xem cùng một họ kiến trúc, việc đổi cấu hình huấn luyện có cải thiện độ ổn định hoặc độ chính xác hay không.

### 6) `qwen_3b_10epochs_grammar_FIX_B3.ipynb`
**Vai trò:** xử lý thêm vấn đề mất cân bằng phân phối band điểm.

**Phương pháp / hướng tiếp cận:**
- Vẫn là **Qwen + grammar features + multi-task regression**.
- Bổ sung hàm `build_band_value_weights(...)` để tạo **trọng số theo từng mức band score**.
- Sinh `sample_weights` cho từng mẫu huấn luyện, dựa trên band của từng tiêu chí.
- Mục tiêu là **sample re-weighting**: các mức band ít gặp sẽ được tăng ảnh hưởng trong loss.

**Ý tưởng cốt lõi:**
- Nếu dữ liệu IELTS bị lệch về vài mức điểm phổ biến, mô hình regression thường học thiên về vùng đó.
- Re-weighting giúp giảm bias phân phối và hỗ trợ mô hình học tốt hơn ở các band hiếm.

### 7) `qwen_3b_10epochs_grammar_FIX_B4.ipynb`
**Vai trò:** đổi bài toán từ regression sang **ordinal regression**.

**Phương pháp / hướng tiếp cận:**
- Thay vì dự đoán trực tiếp band score liên tục, notebook này mã hóa điểm thành **các ngưỡng thứ bậc (ordinal thresholds)**.
- Dùng một kiến trúc như `QwenForIELTSMultiTaskOrdinal`.
- Loss được xây dựng theo kiểu **BCE / threshold-based ordinal loss** trên từng tiêu chí.
- Kết hợp thêm feature theo tiêu chí:
  - `tr_features`
  - `cc_features`
  - `lr_features`
  - `gra_features`
- Tức là mô hình không chỉ có grammar features, mà đã đi tới hướng **criterion-specific handcrafted features** cho cả 4 tiêu chí.

**Ý tưởng cốt lõi:**
- IELTS band score có bản chất **thứ bậc**, không hoàn toàn là giá trị liên tục tự do.
- Ordinal regression thường phù hợp hơn regression thuần khi nhãn có cấu trúc thứ tự rõ ràng.

### 8) `qwen_3b_10epochs_grammar_FIX_B5.ipynb`
**Vai trò:** tiếp tục nhánh ordinal regression, nhưng đổi cấu hình huấn luyện để thực nghiệm hiệu quả batch tốt hơn.

**Phương pháp / hướng tiếp cận:**
- Cùng họ kiến trúc với B4: **ordinal regression + feature theo từng tiêu chí**.
- Dùng batch size/gradient accumulation khác (`BATCH_SIZE = 2` trong cấu hình notebook) để kiểm tra trade-off giữa ổn định gradient và tài nguyên.
- Giữ định hướng hybrid giữa biểu diễn Qwen và feature thủ công.

**Ý nghĩa thực nghiệm:**
- Đây là bản tinh chỉnh cấu hình, giúp so sánh xem thay đổi scheduling/batch có cải thiện được hiệu năng hay độ mượt của quá trình train không.

### 9) `qwen_3b_10epochs_grammar_FIX_B7.ipynb`
**Vai trò:** mở rộng feature engineering mạnh hơn và thay prompt huấn luyện theo hướng sát tác vụ chấm IELTS hơn.

**Phương pháp / hướng tiếp cận:**
- Tạo tập feature đầy đủ cho 4 tiêu chí:
  - **TR**: mức bám đề, độ phủ từ khóa, tương đồng prompt-essay.
  - **CC**: tín hiệu liên kết/coherence.
  - **LR**: lexical richness / diversity.
  - **GRA**: grammar complexity / correctness.
- Hàm `build_input_text(...)` đưa vào chỉ dẫn kiểu giám khảo IELTS, tức là prompt hóa tác vụ rõ hơn cho Qwen.
- Tăng `MAX_LENGTH` lên **2048**, phù hợp hơn với essay dài.
- Giữ hướng **multi-task hybrid model** giữa LLM embedding và feature thủ công nhiều nhóm.

**Ý tưởng cốt lõi:**
- Không chỉ bổ sung feature, mà còn điều chỉnh cách “đặt bài toán” cho mô hình thông qua instruction-style prompt.
- Đây là hướng tiếp cận gần với **instruction-aware essay scoring** hơn so với các notebook trước.

### 10) `qwen_3b_10epochs_grammar_FIX_B8.ipynb`
**Vai trò:** một bản tối ưu thực dụng hơn của B7 cho throughput huấn luyện.

**Phương pháp / hướng tiếp cận:**
- Vẫn dùng bộ feature đầy đủ theo từng criterion như B7.
- Prompt đầu vào được viết rõ theo vai trò **IELTS Writing Task 2 examiner**.
- `MAX_LENGTH = 1536`, `BATCH_SIZE = 8` trong cấu hình notebook, cho thấy định hướng cân bằng giữa độ dài ngữ cảnh và tốc độ train.
- Tiếp tục huấn luyện theo hướng hybrid giữa backbone Qwen và feature engineered.

**Khi nào nên dùng notebook này:**
- Khi muốn một cấu hình thực nghiệm thực dụng hơn B7.
- Khi cần thử nghiệm tốc độ/throughput tốt hơn mà vẫn giữ nhóm feature đầy đủ.

## 🧠 Tóm tắt nhanh các hướng tiếp cận trong `further_test/`

- **Hướng 1 – Text-only regression:**
  - `qwen_3b_3epochs.ipynb`
  - `qwen_3b_10epochs.ipynb`
- **Hướng 2 – Hybrid regression (Qwen + grammar features):**
  - `qwen_3b_10epochs_grammar.ipynb`
  - `qwen_3b_10epochs_grammar_FIX_B1.ipynb`
  - `qwen_3b_10epochs_grammar_FIX_B2.ipynb`
  - `qwen_3b_10epochs_grammar_FIX_B3.ipynb`
- **Hướng 3 – Hybrid ordinal / criterion-specific features:**
  - `qwen_3b_10epochs_grammar_FIX_B4.ipynb`
  - `qwen_3b_10epochs_grammar_FIX_B5.ipynb`
- **Hướng 4 – Full feature + instruction-aware setup:**
  - `qwen_3b_10epochs_grammar_FIX_B7.ipynb`
  - `qwen_3b_10epochs_grammar_FIX_B8.ipynb`

## ⚡ Hướng dẫn sử dụng nhanh

### 1) Chuẩn bị môi trường

Khuyến nghị Python **3.10+** và Jupyter Notebook/Lab.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jupyter pandas numpy scikit-learn matplotlib seaborn transformers datasets torch peft accelerate
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
4. Notebook trong `further_test/`
5. Notebook trong `inference/`

### 3) Dữ liệu

Repository đã chứa sẵn file train/val/test. Nếu thay dataset:
- giữ nguyên schema cột mà notebook đang dùng,
- cập nhật đường dẫn file trong các cell đọc dữ liệu,
- kiểm tra lại các cột label và các feature thủ công trước khi chạy notebook trong `further_test/`.

## 🧭 Gợi ý workflow tái lập kết quả

1. Chạy EDA để kiểm tra phân phối điểm và outlier.
2. Chốt tập đặc trưng ở `feature engineering.ipynb`.
3. Huấn luyện baseline để lấy mốc so sánh.
4. Chạy dần các notebook Qwen từ đơn giản đến phức tạp.
5. Dùng notebook inference tương ứng với checkpoint đã chọn để đánh giá ngoài tập train/val.

## 🧷 Lưu ý

- ⚠️ Dự án thiên về notebook nên kết quả có thể phụ thuộc môi trường chạy.
- ⚠️ Nhiều notebook được viết theo ngữ cảnh Google Colab (`/content/...`, `drive.mount(...)`), nên khi chạy local cần sửa đường dẫn.
- 📦 Nên bổ sung `requirements.txt` hoặc `environment.yml` nếu muốn tái lập dễ hơn.

---

# English below

# ✍️ IELTS Writing Evals

> 📚 This repository is for **research and evaluation of IELTS Writing essays** using NLP models and LLM-based scoring pipelines. The project follows a notebook-oriented workflow: **EDA → feature engineering → baseline training → further experiments → inference**.

## 🎯 Project goals

- 🔍 Explore the IELTS Writing scoring dataset at criterion level.
- 🧠 Build training and evaluation pipelines for essay-scoring models.
- ⚖️ Compare baseline transformer models with more advanced fine-tuning setups.
- 🚀 Run extended experiments with **Qwen 2.5-3B** under `further_test/`.

## 🗂️ Repository structure

### Main notebooks
- 📊 `eda.ipynb`: Exploratory data analysis.
- 🛠️ `feature engineering.ipynb`: Manual feature creation and preprocessing.

### Baseline (`baseline/`)
- `distil_roberta_base_score.ipynb`: Lightweight baseline.
- `roberta_base_score.ipynb`: Standard RoBERTa baseline.
- `roberta_large_score.ipynb`: Larger RoBERTa variant.
- `modern_bert_large_score.ipynb`: ModernBERT Large baseline.

### Inference (`inference/`)
- `Inference.ipynb`: General inference flow.
- `Inference_grammar_feature.ipynb`: Inference for grammar-feature-aware models.
- `B1_inference.ipynb`, `B2_inference.ipynb`, `B7_inference.ipynb`: Inference notebooks for specific checkpoints.

## 🔬 Detailed explanation of `further_test/`

This folder contains advanced experiments built around **`Qwen/Qwen2.5-3B-Instruct`**. Most notebooks use **LoRA/PEFT**, merge the prompt and essay into one textual input, and predict the **four IELTS Writing criteria**. The notebooks can be read as an evolution path: from plain text regression, to hybrid text + handcrafted features, and then to band-aware reweighting, criterion-specific features, and ordinal regression.

### 1) `qwen_3b_3epochs.ipynb`
**Role:** the simplest Qwen baseline.

**Method / approach:**
- Uses `Qwen2.5-3B-Instruct` as the backbone.
- Frames the task as **multi-output regression** with four outputs.
- Builds a single input from prompt + essay.
- Fine-tunes with **LoRA** for efficiency.
- Uses the standard Hugging Face `Trainer`.
- Trains for **3 epochs** as a quick sanity-check baseline.

**Best use case:**
- Establishing a clean starting point.
- Verifying that data loading, tokenization, metrics, and training all work before trying more complex variants.

### 2) `qwen_3b_10epochs.ipynb`
**Role:** longer training with criterion-aware loss weighting.

**Method / approach:**
- Still a **4-output regression** model.
- Extends training to **10 epochs**.
- Introduces a custom `WeightedLossTrainer` so different scoring criteria can contribute different loss weights.
- Still text-only, without handcrafted grammar features.

**Core idea:**
- Some criteria may be harder to learn or more important to optimize, so per-criterion loss weighting may improve training behavior.

### 3) `qwen_3b_10epochs_grammar.ipynb`
**Role:** first hybrid model that injects handcrafted grammar signals.

**Method / approach:**
- Extracts **grammar-related features** from the essay.
- Uses a **hybrid / multi-task setup**:
  - one branch from Qwen text embeddings,
  - one branch from `gra_features`.
- Uses a custom `IELTSMultiTaskTrainer` with multiple loss components.
- Adds a bias/auxiliary loss term for regularization.

**Core idea:**
- Essay quality is not only semantic; surface linguistic indicators also matter, especially for grammar-related scoring.

### 4) `qwen_3b_10epochs_grammar_FIX_B1.ipynb`
**Role:** a stabilized grammar-aware variant for the B1 checkpoint line.

**Method / approach:**
- Keeps the **Qwen + grammar features + multi-task regression** design.
- Retains the custom trainer and multi-loss setup.
- Adjusts training weights/hyperparameters for a more stable B1-oriented checkpoint.
- Appears designed to support lighter downstream inference packaging.

### 5) `qwen_3b_10epochs_grammar_FIX_B2.ipynb`
**Role:** B2 variant of the grammar-aware family.

**Method / approach:**
- Keeps the **hybrid text + grammar feature** formulation.
- Remains in **multi-output regression**, not ordinal modeling.
- Tweaks optimization settings such as batch/gradient accumulation/loss balance for this experiment line.

### 6) `qwen_3b_10epochs_grammar_FIX_B3.ipynb`
**Role:** adds band-aware reweighting to reduce score-distribution bias.

**Method / approach:**
- Still uses **Qwen + grammar features + multi-task regression**.
- Adds `build_band_value_weights(...)` to assign **weights to score bands**.
- Builds per-sample `sample_weights` based on criterion band values.
- Uses **sample reweighting** to make underrepresented bands matter more during training.

**Core idea:**
- If the training set is skewed toward common score ranges, the model may overfit to those ranges; reweighting helps counter that imbalance.

### 7) `qwen_3b_10epochs_grammar_FIX_B4.ipynb`
**Role:** switches from continuous regression to **ordinal regression**.

**Method / approach:**
- Treats score prediction as ordered thresholds rather than free-form continuous outputs.
- Uses a model variant such as `QwenForIELTSMultiTaskOrdinal`.
- Optimizes a **BCE / threshold-based ordinal loss**.
- Incorporates criterion-specific handcrafted features for:
  - `tr_features`
  - `cc_features`
  - `lr_features`
  - `gra_features`

**Core idea:**
- IELTS band scores are inherently ordered labels, so ordinal regression can be more structurally appropriate than plain regression.

### 8) `qwen_3b_10epochs_grammar_FIX_B5.ipynb`
**Role:** an ordinal-regression follow-up with different training configuration.

**Method / approach:**
- Same broad family as B4: **ordinal regression + criterion-specific features**.
- Uses a different batching setup (the notebook config sets `BATCH_SIZE = 2`) to test optimization stability and resource trade-offs.
- Keeps the hybrid Qwen + engineered-feature design.

### 9) `qwen_3b_10epochs_grammar_FIX_B7.ipynb`
**Role:** expands feature engineering and makes the textual prompt more examiner-like.

**Method / approach:**
- Builds a richer feature set for all four criteria:
  - **TR**: prompt adherence / topical coverage,
  - **CC**: coherence and cohesion cues,
  - **LR**: lexical richness,
  - **GRA**: grammar complexity/correctness.
- Rewrites `build_input_text(...)` in a more instruction-like examiner style.
- Increases `MAX_LENGTH` to **2048** for longer essays.
- Continues the **hybrid multi-task** direction with LLM embeddings plus handcrafted features.

**Core idea:**
- This notebook improves both the feature side and the prompting side, making the setup more aligned with instruction-aware essay scoring.

### 10) `qwen_3b_10epochs_grammar_FIX_B8.ipynb`
**Role:** a more throughput-oriented refinement of B7.

**Method / approach:**
- Keeps the full criterion-specific feature set.
- Uses an input prompt written explicitly from the perspective of an **IELTS Writing Task 2 examiner**.
- Uses `MAX_LENGTH = 1536` and `BATCH_SIZE = 8`, suggesting a practical balance between context length and training speed.
- Preserves the hybrid Qwen + engineered-feature training direction.

**Best use case:**
- When you want a more practical high-throughput experiment while keeping the full handcrafted feature pipeline.

## 🧠 Quick taxonomy of `further_test/`

- **Approach 1 – Text-only regression:**
  - `qwen_3b_3epochs.ipynb`
  - `qwen_3b_10epochs.ipynb`
- **Approach 2 – Hybrid regression (Qwen + grammar features):**
  - `qwen_3b_10epochs_grammar.ipynb`
  - `qwen_3b_10epochs_grammar_FIX_B1.ipynb`
  - `qwen_3b_10epochs_grammar_FIX_B2.ipynb`
  - `qwen_3b_10epochs_grammar_FIX_B3.ipynb`
- **Approach 3 – Hybrid ordinal / criterion-specific features:**
  - `qwen_3b_10epochs_grammar_FIX_B4.ipynb`
  - `qwen_3b_10epochs_grammar_FIX_B5.ipynb`
- **Approach 4 – Full feature engineering + instruction-aware setup:**
  - `qwen_3b_10epochs_grammar_FIX_B7.ipynb`
  - `qwen_3b_10epochs_grammar_FIX_B8.ipynb`

## ⚡ Quick start

### 1) Environment setup

Recommended: Python **3.10+** with Jupyter Notebook/Lab.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jupyter pandas numpy scikit-learn matplotlib seaborn transformers datasets torch peft accelerate
```

### 2) Run notebooks

```bash
jupyter notebook
```

Recommended order:
1. `eda.ipynb`
2. `feature engineering.ipynb`
3. notebooks in `baseline/`
4. notebooks in `further_test/`
5. notebooks in `inference/`

### 3) Data

The repository already includes train/validation/test CSV files. If you replace the dataset:
- keep the same column schema used by the notebooks,
- update paths in the data-loading cells,
- verify label columns and handcrafted feature columns before running notebooks in `further_test/`.

## 🧷 Notes

- ⚠️ This is a notebook-heavy repository, so results may depend on the execution environment.
- ⚠️ Several notebooks were written with Google Colab assumptions (`/content/...`, `drive.mount(...)`), so local execution may require path edits.
- 📦 Adding a `requirements.txt` or `environment.yml` would improve reproducibility.
