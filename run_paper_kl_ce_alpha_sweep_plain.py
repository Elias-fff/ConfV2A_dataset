import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]


def alpha_dir_name(alpha):
    return f"alpha_{alpha:.2f}".replace(".", "_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run plain KL+CE distillation once per alpha and summarize results."
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Alpha/lambda values to test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("distill_outputs_paper_kl_ce_alpha_sweep_plain_old"),
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("train_audio_student_distill_paper_kl_ce.py"),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an alpha directory if distill_run_meta.json already exists.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the training script.",
    )
    return parser.parse_args()


def read_result(run_dir):
    meta_path = run_dir / "distill_run_meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return {
        "alpha": float(meta["alpha"]),
        "student_test_loss": float(meta["student_test_loss"]),
        "student_test_accuracy": float(meta["student_test_accuracy"]),
        "teacher_test_accuracy": float(meta["teacher_test_accuracy"]),
        "output_dir": str(run_dir),
    }


def write_summary(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: row["alpha"])

    csv_path = output_dir / "alpha_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "alpha",
                "student_test_loss",
                "student_test_accuracy",
                "teacher_test_accuracy",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["student_test_accuracy"])
    with (output_dir / "best_alpha.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, ensure_ascii=False, indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot(
        [row["alpha"] for row in rows],
        [row["student_test_accuracy"] for row in rows],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("lambda / alpha")
    plt.ylabel("Accuracy")
    plt.title("KL+CE distillation alpha sweep")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "alpha_sweep_accuracy.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for alpha in args.alphas:
        run_dir = args.output_dir / alpha_dir_name(alpha)
        meta_path = run_dir / "distill_run_meta.json"
        if args.skip_existing and meta_path.exists():
            print(f"Reusing {run_dir}")
        else:
            command = [
                args.python,
                str(args.train_script),
                "--alpha",
                str(alpha),
                "--output-dir",
                str(run_dir),
            ]
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)
        rows.append(read_result(run_dir))
        write_summary(rows, args.output_dir)

    print(f"Wrote {args.output_dir / 'alpha_sweep_summary.csv'}")
    print(f"Wrote {args.output_dir / 'alpha_sweep_accuracy.png'}")


if __name__ == "__main__":
    main()
