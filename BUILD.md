# 🛠️ BUILD.md – Dunedain: Fine-Tuning a Domain-Specific LLM

Fine-tunes the `unsloth/Llama-3.2-1B` model on U.S. Army FM 5-0 doctrine using LoRA adapters and includes optional semantic search over source material.

---

## 📌 Project Summary

This pipeline includes:

- Parsing the FM 5-0 Army Field Manual  
- Generating QA examples via ChatGPT  
- Fine-tuning a LLaMA 3.2-1B model using LoRA  
- Evaluating fine-tuned vs. base model responses  
- *(Optional)* Performing semantic vector search over the document

---

## ⚙️ Environment Setup

### 1. Launch EC2 (GPU-backed instance)

- **Instance type:** `g5.xlarge`  
- **OS:** Ubuntu 22.04+  
- **GPU:** NVIDIA A10G  

---

### 2. Install system dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv build-essential
```

---

### 3. Set up Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Python requirements

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```txt
torch
transformers>=4.40.0
datasets
peft
accelerate
trl
sentence-transformers
pymupdf
unsloth
```

---

### 5. Install GPU driver (if needed)

```bash
sudo apt install nvidia-driver-535
sudo reboot
```

After reboot, verify driver install with:

```bash
nvidia-smi
```

---

## 🚀 Project Steps

### 🔹 Step 1. Parse FM 5-0 PDF into chunks

```bash
python parse_pdf.py
```

- Uses PyMuPDF to extract and segment raw text  
- **Output:** `fm5_chunks.json`

---

### 🔹 Step 2. Generate QA training data with ChatGPT 4o mini

```bash
python generate_training_data.py
```

- Uses OpenAI API to create ~500 question–answer pairs from `fm5_chunks.json`  
- **Output:** `train_data.jsonl`  
- Format:

```json
{"question": "What is mission command?", "answer": "Mission command is the exercise of authority..."}
```

---

### 🔹 Step 3. Convert to Unsloth format

```bash
python convert_to_unsloth_format.py
```

- Converts QA into instruction/output format  
- **Output:** `unsloth_dataset.jsonl`

---

### 🔹 Step 4. Fine-tune the model

```bash
python finetune_unsloth.py
```

- **Base model:** `unsloth/Llama-3.2-1B`  
- **LoRA rank:** 16  
- **Epochs:** 3  
- **Output:** `finetuned_llama3_1b/`

---

### 🔹 Step 5. Evaluate model performance

```bash
python evaluate_models.py
```

- Compare generations from base and fine-tuned model  
- Uses a handful of domain-specific prompts

---

### 🔹 Step 6 (Optional). Semantic vector search

```bash
python search_chunks.py
```

- Embeds document using `sentence-transformers`  
- Enables natural-language search over FM 5-0 content
