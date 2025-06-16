# 🧠 Domain-Specific Fine-Tuning with Unsloth

Fine-tune a lightweight `unsloth/Llama-3.2-1B` model on military doctrine (FM 5-0). This project covers PDF parsing, training data generation using ChatGPT 4o mini, instruction-tuning via Unsloth + LoRA, and evaluation—including a semantic search demo.

---

## 🚀 Quickstart

```bash
# 1. Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Parse and chunk FM 5-0 PDF
python parse_pdf.py

# 3. Generate training data (ChatGPT 4o mini)
python generate_training_data.py

# 4. Convert to Unsloth format & fine-tune
python convert_to_unsloth_format.py
python finetune_unsloth.py

# 5. Evaluate base vs. fine-tuned model
python evaluate_models.py

# 6. Bonus: semantic search over FM 5-0
python search_chunks.py
```

---

## 📂 Project Files

- `parse_pdf.py` — Extracts and chunks text from the FM 5-0 PDF  
- `generate_training_data.py` — Uses OpenAI API (ChatGPT 4o mini) to generate QA-style examples  
- `train_data.jsonl` — QA-style examples generated via `generate_training_data.py`  
- `convert_to_unsloth_format.py` — Converts QA pairs into Unsloth fine-tuning format  
- `finetune_unsloth.py` — Fine-tunes the model using LoRA + Hugging Face Transformers  
- `evaluate_models.py` — Compares outputs from base and fine-tuned models  
- `search_chunks.py` — Bonus: semantic search over FM 5-0 using vector embeddings  
- `fm5_chunks.json` — Parsed and structured FM 5-0 document chunks  
- `unsloth_dataset.jsonl` — Final dataset formatted for LoRA fine-tuning  
- `Build And Issues Log.pdf` — Setup process, bugs, and troubleshooting log  
- `Math Foundations.pdf` — LaTeX write-up of mathematical methods used in training or evaluation

> **Note:** Fine-tuned model weights (e.g., adapter files) are *not included* in this repository to keep it lightweight. The model can be re-trained by running `finetune_unsloth.py`.

---

## 🧾 Data Artifacts

- `fm5_chunks.json` — Parsed FM 5-0 text, chunked for QA generation  
- `train_data.jsonl` — Initial QA-style training examples (ChatGPT 4o mini)  
- `unsloth_dataset.jsonl` — Final data formatted for LoRA fine-tuning
