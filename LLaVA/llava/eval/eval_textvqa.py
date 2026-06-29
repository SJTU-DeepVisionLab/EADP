import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--visual_token_num', type=int, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))['data']
    annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": annotation['answers'],
        })

    evaluator = TextVQAAccuracyEvaluator()

    accuracy = evaluator.eval_pred_list(pred_list)
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * accuracy))

    if args.visual_token_num is not None:
        import datetime
        # json is already imported at module level
        log_file = os.path.join(os.path.dirname(annotation_file), "eval_results.jsonl")
        record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "visual_token_num": args.visual_token_num,
            "beta": args.beta,
            "alpha": args.alpha,
            "metrics": {"accuracy": f"{100. * accuracy:.2f}%"}
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Result saved to {log_file}")


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))
