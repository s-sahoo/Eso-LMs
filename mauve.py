import argparse
import json
import pathlib
import sys
sys.path.append("/mnt/weka/home/zhihan.yang/Eso-LMs")
import custom_mauve.src.mauve as mauve


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

    human_references_list = load_json(args.human_references_path)
    samples_list = load_json(args.samples_path)['generated_seqs']

    max_text_length = 512
    results = mauve.compute_mauve(
        p_text=human_references_list,
        q_text=samples_list,
        device_id=0,
        max_text_length = max_text_length,
        verbose=False,
        featurize_model_name="answerdotai/ModernBERT-large",
        featurize_model_type="bert"
    )
    mauve_score = results.mauve

    (args.log_dir / "metric.txt").write_text(str(mauve_score), encoding="utf-8")


if __name__ == "__main__":
    main()
