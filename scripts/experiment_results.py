import pandas as pd
from IPython.display import display
from pathlib import Path

def model_training_results(ROOT, filename, cv_folds):
    """
    Function provides the results for model training
    """
    results_dir = ROOT / "outputs" / filename
    histories = {}

    for fold in cv_folds:
        history = pd.read_csv(results_dir / f"fold_{fold}" / "history.csv")
        histories[fold] = history

        print(f"\nFold {fold}")
        display(history)

    results = (pd.read_csv(results_dir / "fold_results.csv").sort_values("fold").reset_index(drop=True))
    results[["fold", "epoch"]] = (results[["fold", "epoch"]].astype(int))

    print("\nBest results per fold")
    display(results)

    print("\nMetrics mean and std across folds")
    display(results[["mAP", "AP50", "AP_small"]].agg(["mean", "std"]))

    return results


def model_training_results_yolo_cv(ROOT, filename):
    results_path = Path(ROOT) / "outputs" / filename / "fold_results.csv"
    results = pd.read_csv(results_path).sort_values("fold").reset_index(drop=True)
    results[["fold", "epoch"]] = results[["fold", "epoch"]].astype(int)

    print("Final validation results per fold")
    display(results)

    print("\nMetrics mean and std across folds")
    display(results[["mAP", "AP50", "AP_small"]].agg(["mean", "std"]))

    return results

def model_training_results_final(run_dir):
    run_dir = Path(run_dir)
    history = pd.read_csv(run_dir / "history.csv")

    print("Training history")
    display(history)

    best = history.loc[history["mAP"].idxmax()]
    results = best[["mAP", "AP50", "AP_small"]].to_frame().T

    print(f"\nBest validation results — epoch {int(best['epoch'])}")
    display(results)

    return results
