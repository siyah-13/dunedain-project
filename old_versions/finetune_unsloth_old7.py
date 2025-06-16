from unsloth import FastLanguageModel, SFTTrainer
from datasets import load_dataset
import torch

# Load dataset
data = load_dataset("json", data_files="unsloth_dataset.jsonl", split="train")

# Load model & tokenizer
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

# Formatting function
def formatting_func(example):
    return f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}"

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data,
    formatting_func=formatting_func,
    max_seq_length=512,
    packing=False,
    batch_size=2,
    gradient_accumulation_steps=4,
    epochs=3,
    lr=2e-4,
)

# Train the model
trainer.train()

# Save the model
trainer.save("finetuned_llama3_1b")
