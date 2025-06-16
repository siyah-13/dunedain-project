from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# 🧪 Define test prompts
test_prompts = [
    "What is the purpose of the FM 5-0 manual?",
    "How does mission command relate to operations?",
    "Explain the role of planning in the military decision-making process.",
]

# 🔧 Shared model parameters
MAX_SEQ_LENGTH = 512
DTYPE = torch.bfloat16

print("🔄 Loading base model...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=False,
)

print("🔄 Loading fine-tuned model...")
finetuned_model, finetuned_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=False,
)
finetuned_model = PeftModel.from_pretrained(finetuned_model, "./finetuned_llama3_1b")

# 🔎 Helper for generating answers
def generate_response(model, tokenizer, prompt):
    input_text = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()

# 📊 Compare outputs
for prompt in test_prompts:
    print("=" * 80)
    print(f"🧠 Prompt: {prompt}")
    
    base_response = generate_response(base_model, base_tokenizer, prompt)
    print("\n📦 Base Model:\n", base_response)
    
    finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt)
    print("\n🎯 Fine-Tuned Model:\n", finetuned_response)
