import pandas as pd
from IPython.display import display

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
