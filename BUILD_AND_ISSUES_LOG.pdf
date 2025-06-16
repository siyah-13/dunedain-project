# 🛠️ Build and Issues Log

## 📊 Project Setup

**Environment:** Amazon EC2  
**Instance Type:** Started with `t3.large` (CPU), upgraded to `g5.xlarge` (GPU)  
**OS:** Ubuntu 24.04  
**GPU:** NVIDIA A10G (driver manually installed)

### 🐍 Python Environment

```bash
sudo apt update && sudo apt install python3-venv python3-pip -y
python3 -m venv venv
source venv/bin/activate
```

### 📦 Core Libraries Installed

```bash
pip install torch transformers datasets peft accelerate trl \
            sentence-transformers pymupdf unsloth
```

---

## 📁 Final Project Structure

```
dunedain-project/
├── parse_pdf.py
├── convert_to_unsloth_format.py
├── train_data.jsonl
├── unsloth_dataset.jsonl
├── finetune_unsloth.py
├── evaluate_models.py
├── search_chunks.py
├── fm5_chunks.json
├── finetuned_llama3_1b/           # adapter weights & tokenizer
├── requirements.txt
├── README.md
├── BUILD.md
├── .gitignore
└── old_versions/                  # archived experiments
```

---

## 🚨 Issue Log

### ❌ `FastLanguageModel.train()` Not Found

- **Error:** `AttributeError: FastLanguageModel has no attribute 'train'`
- **Fix:** Switched to `HuggingFace Trainer` API for fine-tuning

### ❌ Tensor Shape Error During Tokenization

- **Error:** `ValueError: expected sequence of length X at dim 1 (got Y)`
- **Cause:** Inconsistent input lengths
- **Fix:** Used `padding='max_length'` and `truncation=True` in tokenizer

### ❌ `adapter_config.json` Not Found

- **Error:** LoRA adapter files missing after training
- **Fix:** Saved model via `model.save_pretrained("finetuned_llama3_1b")`

### ❌ `nvidia-smi` Missing

- **Cause:** Instance launched was CPU-only
- **Fix:**
  - Upgraded to `g5.xlarge`
  - Installed driver via `sudo apt install nvidia-driver-535`
  - Rebooted and verified with `nvidia-smi`

### ❌ `ModuleNotFoundError: No module named 'fitz'`

- **Cause:** `pymupdf` not installed
- **Fix:** Installed with `pip install pymupdf`

### ❌ Hugging Face Model Path Issues

- **Cause:** `PeftModel.from_pretrained()` expected model repo ID, not local path
- **Fix:** Ensured `adapter_config.json` and `adapter_model.bin` exist in the target directory and used the correct relative path
