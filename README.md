# 🧠 Domain-Specific Fine-Tuning with Unsloth

Fine-tune a lightweight `unsloth/Llama-3.2-1B` model on military doctrine (FM 5-0). Includes training, evaluation, and semantic search.

---

## 🚀 Quickstart

```bash
# 1. Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Parse and chunk FM 5-0 PDF
python parse_pdf.py

# 3. Generate training data & fine-tune
python convert_to_unsloth_format.py
python finetune_unsloth.py

# 4. Evaluate base vs. fine-tuned model
python evaluate_models.py

# 5.  Semantic search with vector similarity
python search_chunks.py
```

---

## 📂 Project Files

- `parse_pdf.py` — Extracts and chunks text from the FM 5-0 PDF
- `train_data.jsonl` — QA-style training examples (generated via ChatGPT)
- `convert_to_unsloth_format.py` — Converts QA pairs into Unsloth fine-tuning format
- `finetune_unsloth.py` — Fine-tunes the model using LoRA + Hugging Face Transformers
- `evaluate_models.py` — Compares outputs from base and fine-tuned models
- `search_chunks.py` — Bonus: semantic search over FM 5-0 using vector embeddings
- `fm5_chunks.json` — Parsed and structured FM 5-0 document chunks
- `unsloth_dataset.jsonl` — Final dataset formatted for LoRA fine-tuning
- `finetuned_llama3_1b/` — Output directory containing adapter weights + tokenizer
- `Build And Issues Log.pdf` — Documentation of setup process, bugs, and resolution steps
- `Math Foundations.pdf` — LaTeX write-up of mathematical methods used in training or evaluation

### 🧾 Data Artifacts

- `fm5_chunks.json`: Parsed FM 5-0 text, chunked for QA
- `train_data.jsonl`: ChatGPT-labeled Q&A pairs
- `unsloth_dataset.jsonl`: Final fine-tuning dataset
- `finetuned_llama3_1b/`: Output folder with model + adapter weights

---


