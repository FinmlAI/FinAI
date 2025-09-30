import json

def convert_jsonl_to_json(jsonl_file, json_file):
    json_objects = []

    with open(jsonl_file, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                obj = json.loads(line)
                json_objects.append(obj)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid JSON on line {line_number}: {e}")

    with open(json_file, 'w', encoding='utf-8') as outfile:
        json.dump(json_objects, outfile, indent=4)

    print(f"Converted {len(json_objects)} records from {jsonl_file} to {json_file}")


# Example usage
if __name__ == "__main__":
    convert_jsonl_to_json('synthetic_credit_data (2).jsonl', 'input.json')
