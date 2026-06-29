import os
import argparse
import json
import datetime
from llava.eval.m4c_evaluator import EvalAIAnswerProcessor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, required=True)
    parser.add_argument('--result-file', type=str, required=True)
    parser.add_argument('--visual_token_num', type=int, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load GT
    # VizWiz val.json is expected to be a list of entries
    print(f"Loading annotations from {args.annotation_file}...")
    with open(args.annotation_file, 'r') as f:
        gt_data = json.load(f)
    
    # Load Predictions (JSONL)
    # Expected format: {"question_id": 0, "text": "prediction", ...}
    print(f"Loading results from {args.result_file}...")
    preds = {}
    with open(args.result_file, 'r') as f:
        for line in f:
            res = json.loads(line)
            preds[res['question_id']] = res['text']
            
    processor = EvalAIAnswerProcessor()
    
    total_acc = 0
    count = 0
    missing = 0
    
    for i, item in enumerate(gt_data):
        # We assume question_id is the index, matching convert_vizwiz_val_to_llava.py
        qid = i
        
        if qid not in preds:
            missing += 1
            continue
            
        pred_raw = preds[qid]
        pred_norm = processor(pred_raw)
        
        # Extract GT answers
        # VizWiz 'answers' is a list of dicts: [{'answer': 'foo', ...}, ...]
        gt_answers_raw = [ans['answer'] for ans in item['answers']]
        gt_answers_norm = [processor(ans) for ans in gt_answers_raw]
        
        # Calculate accuracy
        match_count = 0
        for gt in gt_answers_norm:
            if pred_norm == gt:
                match_count += 1
        
        acc = min(1.0, match_count / 3.0)
        total_acc += acc
        count += 1
    
    if count == 0:
        print("No samples evaluated.")
    else:
        print(f"Evaluated {count} samples.")
        print(f"Missing predictions: {missing}")
        acc_percent = total_acc / count * 100
        print(f"Accuracy: {acc_percent:.2f}%")

        if args.visual_token_num is not None:
            log_file = os.path.join(os.path.dirname(args.annotation_file), "eval_results.jsonl")
            record = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "visual_token_num": args.visual_token_num,
                "beta": args.beta,
                "alpha": args.alpha,
                "metrics": {"accuracy": acc_percent / 100}
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"Result saved to {log_file}")

if __name__ == "__main__":
    main()
