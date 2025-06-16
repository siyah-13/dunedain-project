from unsloth import FastLanguageModel
from datasets import load_dataset
import torch

# Load dataset
data = load_dataset("json", data_files="unsloth_dataset.jsonl", split="train")

# Load model
#model, tokenizer = FastLanguageModel.from_pretrained(
#    model_name="unsloth/Llama-3.2-1B",
#    max_seq_length=512,
#    dtype=None,
#    load_in_4bit=True,  # set to False if no GPU
#)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,  # ✅ this avoids quantization and GPU issues
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

# Format dataset for SFT
def formatting_func(example):
    return f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}"

# Train
#model.train(
#    dataset=data,
#    tokenizer=tokenizer,
#    formatting_func=formatting_func,
#    max_seq_length=512,
#    packing=False,
#    batch_size=2,
#    gradient_accumulation_steps=4,
#    epochs=3,
#    lr=2e-4,
#    save_path="finetuned_llama3_1b"
#)

model.train(
    dataset=data,
    tokenizer=tokenizer,
    formatting_func=formatting_func,
    max_seq_length=512,
    packing=False,
    batch_size=2,
    gradient_accumulation_steps=4,
    epochs=3,
    lr=2e-4,
    save_path="finetuned_llama3_1b"
)
