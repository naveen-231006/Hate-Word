"""
run_all.py — Master script to run the entire multi-language pipeline
=====================================================================
Runs all steps sequentially on RTX 3050.

Usage:
    python run_all.py                    # Run everything
    python run_all.py --step preprocess  # Just preprocessing
    python run_all.py --step train       # Just training
    python run_all.py --step evaluate    # Just evaluation
"""

import subprocess
import sys
import time
import argparse


def run(cmd, desc):
    """Run a command and print timing."""
    print(f"\n{'#'*60}")
    print(f"  {desc}")
    print(f"  CMD: {cmd}")
    print(f"{'#'*60}\n")
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    status = "✅" if result.returncode == 0 else "❌"
    print(f"\n{status} {desc} — {elapsed/60:.1f} min (exit code {result.returncode})")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate",
                                 "ensemble", "explain", "visualize"])
    parser.add_argument("--experiment", type=str, default="all",
                        help="Which experiment(s) to train")
    parser.add_argument("--model", type=str, default="all",
                        help="Which model(s) to train")
    args = parser.parse_args()

    total_start = time.time()
    steps_run = 0
    steps_ok = 0

    # ── Step 1: Preprocessing ──
    if args.step in ["all", "preprocess"]:
        ok = run(
            f"{sys.executable} 02_preprocessing_multilang.py --lang all",
            "STEP 1: Preprocess all languages"
        )
        steps_run += 1
        steps_ok += ok

    # ── Step 2: Training ──
    if args.step in ["all", "train"]:
        # Train experiments one at a time to manage VRAM
        experiments = [
            "mono_tamil", "mono_malayalam", "mono_kannada",
            "multi_all",
            "cross_ta_ml", "cross_ta_kn", "cross_ml_kn",
        ]

        if args.experiment != "all":
            experiments = [args.experiment]

        for exp in experiments:
            ok = run(
                f"{sys.executable} 03_train_multilang.py --experiment {exp} --model {args.model}",
                f"STEP 2: Train {exp}"
            )
            steps_run += 1
            steps_ok += ok

    # ── Step 3: Evaluation ──
    if args.step in ["all", "evaluate"]:
        ok = run(
            f"{sys.executable} 04_evaluate_multilang.py --experiment {args.experiment}",
            "STEP 3: Evaluate all experiments"
        )
        steps_run += 1
        steps_ok += ok

    # ── Step 4: Ensemble ──
    if args.step in ["all", "ensemble"]:
        ok = run(
            f"{sys.executable} 07_ensemble_multilang.py --experiment {args.experiment}",
            "STEP 4: Ensemble all experiments"
        )
        steps_run += 1
        steps_ok += ok

    # ── Step 5: Explainability ──
    if args.step in ["all", "explain"]:
        ok = run(
            f"{sys.executable} 06_explainability_multilang.py --experiment mono_tamil --model mbert",
            "STEP 5a: LIME explanations (Tamil)"
        )
        steps_run += 1
        steps_ok += ok

        ok = run(
            f"{sys.executable} 06_explainability_multilang.py --experiment mono_malayalam --model mbert",
            "STEP 5b: LIME explanations (Malayalam)"
        )
        steps_run += 1
        steps_ok += ok

        ok = run(
            f"{sys.executable} 06_explainability_multilang.py --experiment mono_kannada --model mbert",
            "STEP 5c: LIME explanations (Kannada)"
        )
        steps_run += 1
        steps_ok += ok

    # ── Step 6: Paper figures ──
    if args.step in ["all", "visualize"]:
        ok = run(
            f"{sys.executable} visualize_results.py",
            "STEP 6: Generate paper figures"
        )
        steps_run += 1
        steps_ok += ok

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Steps: {steps_ok}/{steps_run} successful")
    print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
