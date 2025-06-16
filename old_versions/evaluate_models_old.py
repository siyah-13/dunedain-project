from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import pipeline
import torch

# Load baseline model (no fine-tuning)
baseline_model, baseline_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Load fine-tuned model (base model + adapter weights)
finetuned_model, finetuned_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
finetuned_model = PeftModel.from_pretrained(finetuned_model, "./finetuned_llama3_1b")

# Prepare evaluation prompt
prompt = "<|user|>\nWhat are the key mission objectives outlined in the Army doctrine?\n<|assistant|>"

# Create text generation pipelines
baseline_pipe = pipeline("text-generation", model=baseline_model, tokenizer=baseline_tokenizer)
finetuned_pipe = pipeline("text-generation", model=finetuned_model, tokenizer=finetuned_tokenizer)

# Run both models
print("\n--- Baseline Model Output ---")
baseline_response = baseline_pipe(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
print(baseline_response)

print("\n--- Fine-tuned Model Output ---")
finetuned_response = finetuned_pipe(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
print(finetuned_response)
