import json

data_list = []
with open('synthetic_credit_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data_list.append(json.loads(line))

print(f"Loaded {len(data_list)} records from synthetic_credit_data.jsonl")