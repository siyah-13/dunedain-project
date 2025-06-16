from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import torch

# Load dataset
data = load_dataset("json", data_files="unsloth_dataset.jsonl", split="train")

# Load model + tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# Format function
def formatting_func(example):
    return f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}"

# Apply formatting
data = data.map(lambda example: {"text": formatting_func(example)})

# Tokenize
data = data.map(lambda example: tokenizer(example["text"]), batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="finetuned_llama3_1b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
