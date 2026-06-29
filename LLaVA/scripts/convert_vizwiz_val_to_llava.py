import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Path to VizWiz val.json')
    parser.add_argument('--dst', type=str, required=True, help='Path to output llava_val.jsonl')
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.src, 'r') as f:
        data = json.load(f)
    
    # VizWiz val.json is a list of entries
    # entry: {"image": "VizWiz_val_00000000.jpg", "question": "...", "answers": [...], ...}
    
    # We might need to map image names if they don't match, but usually they do.
    # LLaVA prompt suffix
    prompt_suffix = "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
    
    with open(args.dst, 'w') as f:
        for i, item in enumerate(data):
            # Use index as question_id if not present, but usually we want to keep original ID?
            # VizWiz dataset items don't strictly have a 'question_id' field in the root sometimes, 
            # but let's check. If not, use index. 
            # Wait, VizWiz data usually looks like VQA data.
            # Let's assume we generate a sequential ID matching the index to be safe or use 'image' name hash?
            # In llava_test.jsonl, question_id was 0, 1, 2...
            # So let's use the index i.
            
            question = item['question']
            image_file = item['image']
            
            line = {
                "question_id": i,
                "image": image_file,
                "text": question + prompt_suffix,
                "category": "default"
            }
            
            f.write(json.dumps(line) + "\n")

if __name__ == "__main__":
    main()
