#!/usr/bin/env python3
from pathlib import Path
import shutil


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    target_subpath = Path(
        "INPFormerTrainer_configs_benchmark_inpformer_inpformer_100e_exp1"
    )
    csv_name = "inpformer_m2ad_best.csv"
    log_name = "log_train_Bird.txt"

    only_birds_dirs = [
        p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("only_birds")
    ]

    if not only_birds_dirs:
        print("No directories starting with 'only_birds' found under", base_dir)
        return

    copied_csv = 0
    copied_log = 0

    for ob_dir in sorted(only_birds_dirs):
        exp_dir = ob_dir / target_subpath
        if not exp_dir.exists():
            print(f"Skip {ob_dir.name}: missing {target_subpath}")
            continue

        # Copy CSV
        csv_src = exp_dir / csv_name
        if csv_src.exists():
            csv_dst = results_dir / f"{ob_dir.name}.csv"
            shutil.copy2(csv_src, csv_dst)
            copied_csv += 1
            print(f"CSV  OK: {csv_src} -> {csv_dst}")
        else:
            print(f"CSV  NG: missing {csv_src}")

        # Copy LOG (preserve original extension if any)
        log_src = exp_dir / log_name
        if log_src.exists():
            log_ext = log_src.suffix or ".log"
            log_dst = results_dir / f"{ob_dir.name}{log_ext}"
            shutil.copy2(log_src, log_dst)
            copied_log += 1
            print(f"LOG  OK: {log_src} -> {log_dst}")
        else:
            print(f"LOG  NG: missing {log_src}")

    print(
        f"Done. Copied CSV: {copied_csv}, Copied LOG: {copied_log}. Results dir: {results_dir}"
    )


if __name__ == "__main__":
    main()


