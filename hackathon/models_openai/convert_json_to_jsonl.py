import json

def convert_json_to_jsonl(input_file: str, output_file: str):
    with open(input_file, 'r') as in_file:
        data = json.load(in_file)

    with open(output_file, 'w') as out_file:
        for item in data:
            json.dump(item, out_file)
            out_file.write('\n')

# Example usage:
input_json_file = 'data1000_v2_openai.json'
output_jsonl_file = 'data1000_v2_openai.jsonl'
convert_json_to_jsonl(input_json_file, output_jsonl_file)
