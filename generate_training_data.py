"""
generate_training_data.py

Uses the OpenAI API with ChatGPT 4o mini to generate QA-style training examples
based on parsed FM 5-0 document chunks. Outputs a JSONL file compatible with
instruction-tuning pipelines like Unsloth.

⚠️ Requires a valid OpenAI API key.
"""

import json
import openai
import time
from pathlib import Path

# === CONFIG ===
CHUNK_FILE = "fm5_chunks.json"
OUTPUT_FILE = "train_data.jsonl"
API_KEY = "sk-..."  # 🔒 API key removed for security
MODEL = "gpt-4o"
BATCH_SIZE = 5
LIMIT = 500  # number of training examples to generate

# === SETUP ===
openai.api_key = API_KEY
chunks = json.load(open(CHUNK_FILE))[:LIMIT]  # cap to limit

# === Prompt Template ===
def make_prompt(chunk_text):
    return (
        f"You are an Army field operations assistant. Based only on the following text, "
        f"generate a realistic question a soldier might ask, and a concise, accurate answer.\n"
        f"\nTEXT:\n{chunk_text}\n"
        f"\nFORMAT: Return as JSON: {{\"question\": ..., \"answer\": ...}}"
    )

# === Write JSONL ===
with open(OUTPUT_FILE, "w") as out:
    for i, item in enumerate(chunks):
        prompt = make_prompt(item['text'])
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a military knowledge assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            out.write(json.dumps(parsed) + "\n")
            print(f"[{i+1}/{LIMIT}] ✅ {parsed['question'][:60]}...")
            time.sleep(1.2)  # rate limit buffer
        except Exception as e:
            print(f"[{i+1}] ❌ Error: {e}")
            continue
