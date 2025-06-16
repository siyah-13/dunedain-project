from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import torch

# Step 1: Load dataset
print("Loading dataset...")
data = load_dataset("json", data_files="unsloth_dataset.jsonl", split="train")

# Step 2: Load model + tokenizer
print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Step 3: Apply LoRA adapters
print("Attaching LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# Step 4a: Format dataset
print("Formatting examples...")
def formatting_func(example):
    return f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}"

data = data.map(lambda example: {"text": formatting_func(example)})

# Step 4b: Tokenize and add labels
def tokenize(example):
    tokens = tokenizer(example["text"])
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

data = data.map(tokenize, batched=True)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="finetuned_llama3_1b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch"
)

# Step 6: Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()
print("✅ Done training!")
