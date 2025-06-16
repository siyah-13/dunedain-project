from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import torch

# Step 1: Load dataset
data = load_dataset("json", data_files="unsloth_dataset.jsonl", split="train")

# Step 2: Load base model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Step 3: Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# Step 4: Format examples into instruction-answer format
def formatting_func(example):
    return f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}"

data = data.map(lambda x: {"text": formatting_func(x)})
data = data.map(lambda x: tokenizer(x["text"]), batched=True)

# Step 5: Set up training arguments
training_args = TrainingArguments(
    output_dir="finetuned_llama3_1b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
)

# Step 6: Train using HF Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    tokenizer=tokenizer,
)

trainer.train()

# ✅ Step 7: Save the LoRA adapter and tokenizer
model.save_pretrained("finetuned_llama3_1b")
tokenizer.save_pretrained("finetuned_llama3_1b")
