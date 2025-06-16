from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import torch

# Step 1: Load dataset
print("Loading dataset...")
data = load_dataset("json", data_files="unsloth_dataset.jsonl", split="train")

# Step 2: Load base model
print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Step 3: Add LoRA adapters
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

# Step 4: Format and tokenize examples
print("Formatting examples...")
def format_and_tokenize(example):
    text = f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}"
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_data = data.map(format_and_tokenize, batched=False)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="finetuned_llama3_1b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
)

# Step 6: Set up and run trainer
print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer,
)

trainer.train()

# Step 7: Save final model + tokenizer
model.save_pretrained("finetuned_llama3_1b")
tokenizer.save_pretrained("finetuned_llama3_1b")
print("✅ Training complete and model saved!")
