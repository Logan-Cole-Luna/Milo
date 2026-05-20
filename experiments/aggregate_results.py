import os
import json
import glob
import math
import pandas as pd
import numpy as np
from scipy import stats

"""
Aggregate metrics across experiments and optimizers into CSVs and compute pairwise significance.

Inputs (defaults mirror supervised_learning/config.py):
- base_dir: path to experiments/supervised_learning
- result_dir_name: results or results_nt
- experiments: list[str] of experiment folder names (e.g., ["logistic", "vgg11_cifar10"]). Case-insensitive helper provided.
- optimizers: list[str] optimizer names to include

Artifacts produced under base_dir/aggregate/:
- aggregated_training_metrics.csv
- aggregated_validation_metrics.csv
- aggregated_test_metrics.csv
- pairwise_significance_final_validation_accuracy.csv

Notes:
- Missing files/combinations are filled with NaN.
- Validation metrics come from raw_runs_validation_data_...json and averaged training_metrics.csv when available.
- Test metrics come from *_final_test_results.csv saved by experiment_runner.
"""

# ---------- Helpers ----------

def _safe_read_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _find_first(patterns):
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return matches[0]
    return None


def _canonicalize_name(name: str) -> str:
    return name.strip().lower()


# ---------- Core ----------

def aggregate(
    base_dir: str,
    result_dir_name: str,
    experiments: list,
    optimizers: list,
    output_dir_name: str = "aggregate",
):
    os.makedirs(os.path.join(base_dir, output_dir_name), exist_ok=True)
    out_dir = os.path.join(base_dir, output_dir_name)

    # Containers
    train_rows = []
    val_rows = []
    test_rows = []
    time_rows = []

    # Iterate experiments
    for exp in experiments:
        exp_folder = _canonicalize_name(exp)
        exp_path = os.path.join(base_dir, exp_folder)
        results_path = os.path.join(exp_path, result_dir_name)
        if not os.path.isdir(results_path):
            # Skip if no results folder
            continue

        # Discover file IDs
        # We expect files like:
        # - {exp_id}_training_metrics.csv
        # - raw_runs_validation_data_{exp_id}_validation_curves.json
        # - {exp_id}_final_test_results.csv
        # We can infer exp_id as the part before suffix from any of them
        train_csv = _find_first([
            os.path.join(results_path, f"{exp_folder}_training_metrics.csv"),
            os.path.join(results_path, "*_training_metrics.csv"),
        ])
        raw_val_json = _find_first([
            os.path.join(results_path, f"raw_runs_validation_data_{exp_folder}_validation_curves.json"),
            os.path.join(results_path, "raw_runs_validation_data_*_validation_curves.json"),
        ])
        test_csv = _find_first([
            os.path.join(results_path, f"{exp_folder}_final_test_results.csv"),
            os.path.join(results_path, "*_final_test_results.csv"),
        ])

    # Training metrics (averaged)
        if train_csv and os.path.isfile(train_csv):
            try:
                df_train = pd.read_csv(train_csv)
                df_train["experiment"] = exp
                # Keep only requested optimizers; fill missing combos later via merge
                df_train = df_train[df_train["optimizer"].isin(optimizers)]
                train_rows.append(df_train)
            except Exception:
                pass

    # Validation metrics per-run from raw json
        if raw_val_json and os.path.isfile(raw_val_json):
            data = _safe_read_json(raw_val_json)
            if data:
                # data: { 'val_losses': {opt: [[...run1...], [...run2...]]}, ... }
                for metric_key, pretty in [
                    ("val_losses", "val_loss"),
                    ("val_accuracies", "val_accuracy"),
                    ("val_f1_scores", "val_f1_score"),
                    ("val_aucs", "val_auc"),
                ]:
                    metric_block = data.get(metric_key, {})
                    for opt in optimizers:
                        runs = metric_block.get(opt, []) or []
                        for run_idx, run_series in enumerate(runs, start=1):
                            for epoch_idx, val in enumerate(run_series, start=1):
                                val_rows.append({
                                    "experiment": exp,
                                    "optimizer": opt,
                                    "run": run_idx,
                                    "epoch": epoch_idx,
                                    pretty: val,
                                })

        # Test metrics per-run
        if test_csv and os.path.isfile(test_csv):
            try:
                df_test = pd.read_csv(test_csv)
                df_test["experiment"] = exp
                test_rows.append(df_test)
            except Exception:
                pass

        # Time complexity: derive seconds per epoch from compute_resources CSV (duration_seconds / epochs)
        compute_csv = _find_first([
            os.path.join(results_path, f"compute_resources_{exp_folder}.csv"),
            os.path.join(results_path, "compute_resources_*.csv"),
        ])
        if compute_csv and os.path.isfile(compute_csv):
            try:
                df_comp = pd.read_csv(compute_csv)
                # Expect columns: duration_seconds, optimizer, run_index, epochs, ...
                if "duration_seconds" in df_comp.columns and "epochs" in df_comp.columns and "optimizer" in df_comp.columns:
                    df_comp["experiment"] = exp
                    df_comp["time_per_epoch_sec"] = df_comp["duration_seconds"] / df_comp["epochs"].replace(0, np.nan)
                    # Keep only requested optimizers
                    df_comp = df_comp[df_comp["optimizer"].isin(optimizers)]
                    time_rows.append(df_comp[["experiment", "optimizer", "run_index", "duration_seconds", "epochs", "time_per_epoch_sec"]])
            except Exception:
                pass

    # Build DataFrames
    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(
        columns=["experiment", "optimizer", "epoch", "train_loss", "train_accuracy", "train_f1_score", "train_auc", "train_loss_std_err", "train_accuracy_std_err", "train_f1_score_std_err", "train_auc_std_err"]
    )
    val_df = pd.DataFrame(val_rows) if val_rows else pd.DataFrame(
        columns=["experiment", "optimizer", "run", "epoch", "val_loss", "val_accuracy", "val_f1_score", "val_auc"]
    )
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(
        columns=["experiment", "optimizer", "run", "test_loss", "test_accuracy", "test_f1_score", "test_auc", "test_eval_time_seconds"]
    )
    time_df = pd.concat(time_rows, ignore_index=True) if time_rows else pd.DataFrame(
        columns=["experiment", "optimizer", "run_index", "duration_seconds", "epochs", "time_per_epoch_sec"]
    )

    # Ensure all requested optimizers appear per experiment by padding with NaNs (final epoch/test rows will still be NaN if missing)
    def _pad(df, cols_identity):
        if df.empty:
            return df
        exp_opts = pd.MultiIndex.from_product([
            sorted(set(df["experiment"])) or experiments,
            optimizers,
        ], names=["experiment", "optimizer"])
        # keep any extra keys like epoch/run by merge later
        base = pd.DataFrame(index=exp_opts).reset_index()
        # Use outer merge to include all combinations
        merged = base.merge(df, on=["experiment", "optimizer"], how="left")
        return merged

    train_df = _pad(train_df, ["experiment", "optimizer"]) if not train_df.empty else train_df
    val_df = _pad(val_df, ["experiment", "optimizer"]) if not val_df.empty else val_df
    test_df = _pad(test_df, ["experiment", "optimizer"]) if not test_df.empty else test_df
    time_df = _pad(time_df, ["experiment", "optimizer"]) if not time_df.empty else time_df

    # Save aggregated CSVs
    os.makedirs(out_dir, exist_ok=True)
    train_out = os.path.join(out_dir, "aggregated_training_metrics.csv")
    val_out = os.path.join(out_dir, "aggregated_validation_metrics.csv")
    test_out = os.path.join(out_dir, "aggregated_test_metrics.csv")
    time_out = os.path.join(out_dir, "aggregated_time_complexity.csv")
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)
    # Also compute mean/std_err time per epoch by (experiment, optimizer)
    if not time_df.empty:
        agg = (
            time_df.groupby(["experiment", "optimizer"], dropna=False)["time_per_epoch_sec"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        # std error
        agg["std_err"] = agg.apply(lambda r: (r["std"] / math.sqrt(r["count"])) if r["count"] and r["count"] > 1 else np.nan, axis=1)
        agg.rename(columns={"mean": "mean_time_per_epoch_sec"}, inplace=True)
        agg.to_csv(time_out, index=False)
    else:
        # write empty file with headers
        pd.DataFrame(columns=["experiment", "optimizer", "mean_time_per_epoch_sec", "std_err", "count"]).to_csv(time_out, index=False)

    # Pairwise significance on final validation accuracy (paired across experiments and runs where available)
    sig_mat = _pairwise_significance_final_val(val_df, optimizers)
    sig_out = os.path.join(out_dir, "pairwise_significance_final_validation_accuracy.csv")
    sig_mat.to_csv(sig_out, index=True)

    # --- Extra artifacts: validation mean± across runs (per epoch) ---
    val_mean_out = os.path.join(out_dir, "aggregated_validation_metrics_mean.csv")
    val_final_out = os.path.join(out_dir, "aggregated_validation_final.csv")
    train_final_out = os.path.join(out_dir, "aggregated_training_final.csv")

    try:
        if not val_df.empty:
            # compute per-epoch mean/std_err per (experiment, optimizer, epoch)
            def _mean_stderr(group):
                def _agg(col):
                    arr = group[col].dropna().to_numpy()
                    if arr.size == 0:
                        return np.nan, np.nan
                    mean = float(np.mean(arr))
                    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else np.nan
                    return mean, se
                metrics = ["val_loss", "val_accuracy", "val_f1_score", "val_auc"]
                out = {}
                for m in metrics:
                    mean, se = _agg(m)
                    out[f"{m}_mean"] = mean
                    out[f"{m}_std_err"] = se
                return pd.Series(out)

            group_keys = ["experiment", "optimizer", "epoch"]
            # ensure epoch is numeric for grouping
            if "epoch" in val_df.columns:
                val_df["epoch"] = pd.to_numeric(val_df["epoch"], errors="coerce")
            val_mean = (
                val_df.groupby(group_keys, dropna=False).apply(_mean_stderr).reset_index()
                if not val_df.empty else pd.DataFrame(columns=group_keys + [
                    "val_loss_mean","val_loss_std_err","val_accuracy_mean","val_accuracy_std_err","val_f1_score_mean","val_f1_score_std_err","val_auc_mean","val_auc_std_err"
                ])
            )
            val_mean.to_csv(val_mean_out, index=False)

            # Final-only (last epoch per (experiment, optimizer))
            final_rows = []
            for (exp, opt), g in val_mean.groupby(["experiment", "optimizer"], dropna=False):
                g2 = g.dropna(subset=["epoch"]).sort_values("epoch")
                if g2.empty:
                    continue
                last = g2.iloc[-1]
                final_rows.append({
                    "experiment": exp,
                    "optimizer": opt,
                    "epoch": int(last["epoch"]) if not pd.isna(last["epoch"]) else np.nan,
                    "val_loss_mean": last.get("val_loss_mean", np.nan),
                    "val_loss_std_err": last.get("val_loss_std_err", np.nan),
                    "val_accuracy_mean": last.get("val_accuracy_mean", np.nan),
                    "val_accuracy_std_err": last.get("val_accuracy_std_err", np.nan),
                    "val_f1_score_mean": last.get("val_f1_score_mean", np.nan),
                    "val_f1_score_std_err": last.get("val_f1_score_std_err", np.nan),
                    "val_auc_mean": last.get("val_auc_mean", np.nan),
                    "val_auc_std_err": last.get("val_auc_std_err", np.nan),
                })
            pd.DataFrame(final_rows).to_csv(val_final_out, index=False)
        else:
            pd.DataFrame(columns=["experiment","optimizer","epoch","val_loss_mean","val_loss_std_err","val_accuracy_mean","val_accuracy_std_err","val_f1_score_mean","val_f1_score_std_err","val_auc_mean","val_auc_std_err"]).to_csv(val_mean_out, index=False)
            pd.DataFrame(columns=["experiment","optimizer","epoch","val_loss_mean","val_loss_std_err","val_accuracy_mean","val_accuracy_std_err","val_f1_score_mean","val_f1_score_std_err","val_auc_mean","val_auc_std_err"]).to_csv(val_final_out, index=False)
    except Exception:
        # write placeholders if something goes wrong
        pd.DataFrame(columns=["experiment","optimizer","epoch"]).to_csv(val_mean_out, index=False)
        pd.DataFrame(columns=["experiment","optimizer","epoch"]).to_csv(val_final_out, index=False)

    # --- Final-only for training (mean± already provided per epoch in training CSV via std_err columns) ---
    try:
        if not train_df.empty:
            # There may be multiple rows per (experiment, optimizer, epoch); pick the last epoch per (experiment, optimizer)
            train_df["epoch"] = pd.to_numeric(train_df["epoch"], errors="coerce")
            final_train_rows = []
            for (exp, opt), g in train_df.groupby(["experiment", "optimizer"], dropna=False):
                g2 = g.dropna(subset=["epoch"]).sort_values("epoch")
                if g2.empty:
                    continue
                last = g2.iloc[-1]
                final_train_rows.append({
                    "experiment": exp,
                    "optimizer": opt,
                    "epoch": int(last["epoch"]) if not pd.isna(last["epoch"]) else np.nan,
                    "train_loss": last.get("train_loss", np.nan),
                    "train_loss_std_err": last.get("train_loss_std_err", np.nan),
                    "train_accuracy": last.get("train_accuracy", np.nan),
                    "train_accuracy_std_err": last.get("train_accuracy_std_err", np.nan),
                    "train_f1_score": last.get("train_f1_score", np.nan),
                    "train_f1_score_std_err": last.get("train_f1_score_std_err", np.nan),
                    "train_auc": last.get("train_auc", np.nan),
                    "train_auc_std_err": last.get("train_auc_std_err", np.nan),
                })
            pd.DataFrame(final_train_rows).to_csv(train_final_out, index=False)
        else:
            pd.DataFrame(columns=["experiment","optimizer","epoch","train_loss","train_loss_std_err","train_accuracy","train_accuracy_std_err","train_f1_score","train_f1_score_std_err","train_auc","train_auc_std_err"]).to_csv(train_final_out, index=False)
    except Exception:
        pd.DataFrame(columns=["experiment","optimizer","epoch"]).to_csv(train_final_out, index=False)

    return {
        "train_csv": train_out,
        "val_csv": val_out,
        "test_csv": test_out,
    "sig_csv": sig_out,
    "time_csv": time_out,
        "val_mean_csv": val_mean_out,
        "val_final_csv": val_final_out,
        "train_final_csv": train_final_out,
    }


def _pairwise_significance_final_val(val_df: pd.DataFrame, optimizers: list) -> pd.DataFrame:
    # Compute final epoch per (experiment, run, optimizer)
    if val_df.empty:
        return pd.DataFrame(index=optimizers, columns=optimizers)
    # Find max epoch per experiment/optimizer/run
    grouped = val_df.groupby(["experiment", "optimizer", "run"], dropna=False)
    # Pick last non-null val_accuracy per group
    records = []
    for (exp, opt, run), g in grouped:
        g2 = g.dropna(subset=["val_accuracy"]) if "val_accuracy" in g.columns else g
        if g2 is None or g2.empty:
            continue
        # Identify max epoch row
        try:
            last_row = g2.sort_values("epoch").iloc[-1]
            acc = last_row.get("val_accuracy", np.nan)
            records.append({"experiment": exp, "optimizer": opt, "run": run, "final_val_accuracy": acc})
        except Exception:
            continue
    final_df = pd.DataFrame(records)
    # Build pivot per optimizer with multi-index (experiment, run)
    if final_df.empty:
        return pd.DataFrame(index=optimizers, columns=optimizers)
    pairs = pd.DataFrame(index=optimizers, columns=optimizers)

    # For each pair, create matched samples by inner join on (experiment, run)
    for i, a in enumerate(optimizers):
        for j, b in enumerate(optimizers):
            if a == b:
                pairs.loc[a, b] = 1.0
                continue
            a_df = final_df[final_df["optimizer"] == a][["experiment", "run", "final_val_accuracy"]]
            b_df = final_df[final_df["optimizer"] == b][["experiment", "run", "final_val_accuracy"]]
            merged = pd.merge(a_df, b_df, on=["experiment", "run"], suffixes=("_a", "_b"))
            if len(merged) < 2:
                pairs.loc[a, b] = np.nan
                continue
            diffs = merged["final_val_accuracy_a"] - merged["final_val_accuracy_b"]
            # Wilcoxon signed-rank is robust for paired non-normal; fallback to t-test if needed
            try:
                stat, p = stats.wilcoxon(diffs)
            except Exception:
                _, p = stats.ttest_rel(merged["final_val_accuracy_a"], merged["final_val_accuracy_b"])  # noqa
            pairs.loc[a, b] = float(p)
    return pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate experiment results into CSVs and compute significance.")
    parser.add_argument("--base_dir", type=str, default=os.path.join(os.path.dirname(__file__), "supervised_learning"))
    parser.add_argument("--result_dir_name", type=str, default="results_nt", help="results or results_nt")
    parser.add_argument("--experiments", nargs="*", default=[
        "LOGISTIC", "MULTILAYER", "RESNET34_CIFAR10", "RESNET34_CIFAR100", "VGG11_CIFAR10", "VGG11_CIFAR100", "SIMPLE_VIT", "ATTENTION_CNN"
    ])
    parser.add_argument("--optimizers", nargs="*", default=[
        "MILO", "MILO_LW", "SGD", "ADAMW", "ADAM_MINI", "NOVOGRAD", "ADAGRAD", "ADEMAMIX"
    ])
    args = parser.parse_args()

    # Normalize experiments
    experiments = [str(x).upper() for x in args.experiments]
    # Map symbols to folder names used on disk
    folder_map = {
        "LOGISTIC": "logistic",
        "MULTILAYER": "multilayer",
        "RESNET34_CIFAR10": "resnet34_cifar10",
        "RESNET34_CIFAR100": "resnet34_cifar100",
        "VGG11_CIFAR10": "vgg11_cifar10",
        "VGG11_CIFAR100": "vgg11_cifar100",
        "SIMPLE_VIT": "simple_vit",
        "ATTENTION_CNN": "attention_cnn",
    }
    exp_folders = [folder_map.get(x, x.lower()) for x in experiments]

    out = aggregate(args.base_dir, args.result_dir_name, exp_folders, args.optimizers)
    print("Aggregated CSVs:")
    for k, v in out.items():
        print(f"- {k}: {v}")
