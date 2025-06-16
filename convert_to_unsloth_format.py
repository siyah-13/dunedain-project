import json

input_path = "train_data.jsonl"
output_path = "unsloth_dataset.jsonl"

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        qa = json.loads(line)
        instruction = qa.get("question", "").strip()
        output = qa.get("answer", "").strip()
        if instruction and output:
            record = {
                "instruction": instruction,
                "output": output
            }
            outfile.write(json.dumps(record) + "\n")

print("✅ Done: Saved to unsloth_dataset.jsonl")
