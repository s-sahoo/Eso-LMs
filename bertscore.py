import argparse
import json
import pathlib
import sys
from evaluate import load
import numpy as np


def load_json(path: pathlib.Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        sys.exit(f"[ERROR] {path}: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compute a single scalar metric for two JSON files.")
    p.add_argument('-ref', '--human-references-path', type=pathlib.Path, required=True)
    p.add_argument('-s', '--samples-path', type=pathlib.Path, required=True)
    p.add_argument('-l', '--log-dir', type=pathlib.Path, help="Directory to save metric (metric.txt)", required=True)
    args = p.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)

    human_references_list = load_json(args.human_references_path)[:1000]
    samples_list = load_json(args.samples_path)['generated_seqs'][:1000]
    
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=samples_list, 
                                references=human_references_list, 
                                lang='en',
                                rescale_with_baseline=True)
                                # model_type='microsoft/deberta-xlarge-mnli')

    (args.log_dir / "metric.txt").write_text(
        f"Precision: {np.mean(results['precision'])}, Recall: {np.mean(results['recall'])}, F1: {np.mean(results['f1'])}", 
        encoding="utf-8")


if __name__ == "__main__":
    main()
