"""
ablation.py - Training-epsilon x adversarial-ratio x network-width ablation.

Goal: map the best achievable trade-off between clean detection and robustness,
instead of judging a single hardened model. Every model is evaluated the same
way: a fixed evaluation attack (epsilon 0.3, 40-step constrained PGD) and
thresholds chosen on validation benign flows to hit target false-positive
rates, so models are compared AT THE SAME false-positive rate.

Put this file next to the other scripts (it imports from threshold_analysis.py
and robustness_checks.py) and run, for example:

    python ablation.py --csv "D:\\DL Project\\data\\combined_sample.csv" --seed 0

Smoke test first (a few minutes):

    python ablation.py --csv "D:\\DL Project\\data\\combined_sample.csv" \
        --baseline-epochs 2 --adv-epochs 3 --train-eps 0.3 --adv-ratios 0.5 --widths 128

Time: the default grid trains 3 x 2 x 2 = 12 hardened models plus 2 baselines,
so expect roughly 12 times the adversarial-training time of a single run.
Narrow it with --train-eps, --adv-ratios and --widths.

Width w means hidden layers (w, w/2, w/4): 128 -> (128, 64, 32) is the current
model, 256 -> (256, 128, 64) is a larger one.

Outputs:
  ablation_results.csv  - one row per model x operating point
  ablation_curves.csv   - recall vs false-positive-rate curves
  ablation_curves.png   - clean recall and recall under attack vs FPR
Results are written after every model, so a crash does not lose finished work.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from main import build_schema, FEATURE_NAMES
from data_module import (
    NIDSPreprocessor, make_dataloaders, make_eval_loader, split_validation,
    NIDSClassifier, train_model, compute_class_weights,
)
from attack_module import ConstrainedTabularAttack, AttackConfig
from defense_module import AdvTrainingConfig, adversarial_train
from robustness_checks import craft
from threshold_analysis import malicious_prob, metrics_at, threshold_for_fpr


def hidden_dims_for(width: int):
    return (width, max(width // 2, 8), max(width // 4, 8))


def op_name(target: float) -> str:
    return f"target FPR {target:.1%}"


def collect_probs(model, ctx, device, eval_cfg):
    """Malicious probabilities on validation/test flows, plus attacked malicious flows."""
    model.eval()
    attacker = ConstrainedTabularAttack(model, ctx["scaler"], ctx["mutable"], ctx["integer"],
                                        ctx["nonneg"], device)
    X_adv = craft(attacker, ctx["X_mal"], ctx["y_mal"], eval_cfg)
    return dict(
        ben_val=malicious_prob(model, ctx["X_val_ben"], device),
        ben_test=malicious_prob(model, ctx["X_ben"], device),
        mal_clean=malicious_prob(model, ctx["X_mal"], device),
        mal_adv=malicious_prob(model, X_adv, device),
    )


def add_results(rows, curves, meta, p, targets):
    """Operating-point rows (table) and a dense FPR sweep (curves) for one model."""
    points = [("default 0.5", 0.5)] + [(op_name(t), threshold_for_fpr(p["ben_val"], t)) for t in targets]
    for label, thr in points:
        fpr, rec, rec_adv, esr = metrics_at(thr, p["ben_test"], p["mal_clean"], p["mal_adv"])
        rows.append({**meta, "operating_point": label, "threshold": thr, "test_fpr": fpr,
                     "clean_recall": rec, "recall_after_attack": rec_adv, "corrected_esr": esr})
    for t in np.logspace(-3, np.log10(0.3), 30):
        thr = threshold_for_fpr(p["ben_val"], t)
        fpr, rec, rec_adv, esr = metrics_at(thr, p["ben_test"], p["mal_clean"], p["mal_adv"])
        curves.append({**meta, "test_fpr": fpr, "clean_recall": rec,
                       "recall_after_attack": rec_adv, "corrected_esr": esr})


def save(rows, curves):
    pd.DataFrame(rows).to_csv("ablation_results.csv", index=False)
    pd.DataFrame(curves).to_csv("ablation_curves.csv", index=False)


def plot_curves(curves_df, train_eps_list, ratio_list, width_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(train_eps_list), 1)))
    eps_color = {e: colors[i] for i, e in enumerate(sorted(train_eps_list))}
    styles = ["-", "--", ":", "-."]
    ratio_style = {r: styles[i % len(styles)] for i, r in enumerate(sorted(ratio_list))}
    width_lw = {w: 1.2 + 1.0 * i for i, w in enumerate(sorted(width_list))}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, g in curves_df.groupby("name"):
        first = g.iloc[0]
        if first["kind"] == "baseline":
            kw = dict(color="black", linestyle="-", linewidth=width_lw[first["width"]], alpha=0.8)
        else:
            kw = dict(color=eps_color[first["train_eps"]], linestyle=ratio_style[first["adv_ratio"]],
                      linewidth=width_lw[first["width"]])
        axes[0].plot(g["test_fpr"], g["clean_recall"], label=name, **kw)
        axes[1].plot(g["test_fpr"], g["recall_after_attack"], label=name, **kw)

    for ax, title in zip(axes, ["Clean recall vs false-positive rate",
                                "Recall under attack (eps 0.3) vs false-positive rate"]):
        ax.set_xscale("log")
        ax.set_xlabel("Test false-positive rate")
        ax.set_ylabel("Recall on malicious flows")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("ablation_curves.png", dpi=150)
    print("Saved ablation_results.csv, ablation_curves.csv and ablation_curves.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--adv-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-eps", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--adv-ratios", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--eval-eps", type=float, default=0.3, help="Fixed evaluation attack epsilon")
    parser.add_argument("--targets", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--summary-fpr", type=float, default=0.05,
                        help="False-positive rate used to rank the models in the final table")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] Using device: {device}")

    targets = sorted(set(args.targets + [args.summary_fpr]))

    # ---- Data (same pipeline and split as the other scripts) ----------------
    schema = build_schema()
    pre = NIDSPreprocessor(schema, random_state=args.seed)
    df = pre.load_and_clean(args.csv)
    X_train, X_test, y_train, y_test = pre.fit_transform_split(df)
    X_tr, X_val, y_tr, y_val = split_validation(X_train, y_train, val_frac=0.1, seed=args.seed)
    train_loader, test_loader = make_dataloaders(X_tr, y_tr, X_test, y_test, args.batch_size)
    val_loader = make_eval_loader(X_val, y_val)
    class_weights = compute_class_weights(y_tr)

    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    ctx = dict(
        scaler=pre.scaler,
        mutable=schema.mutable_mask(), integer=schema.integer_mask(), nonneg=schema.non_negative_mask(),
        X_val_ben=X_val_t[y_val_t == 0], X_ben=X_test_t[y_test_t == 0],
        X_mal=X_test_t[y_test_t == 1], y_mal=y_test_t[y_test_t == 1],
    )
    eval_cfg = AttackConfig(epsilon=args.eval_eps, alpha=args.eval_eps / 6, num_steps=40)

    rows, curves = [], []

    # ---- Baselines, one per width -------------------------------------------
    for width in args.widths:
        name = f"baseline w={width}"
        print(f"\n=== {name} ===")
        torch.manual_seed(args.seed)
        model = NIDSClassifier(input_dim=len(FEATURE_NAMES), hidden_dims=hidden_dims_for(width))
        model = train_model(model, train_loader, device, epochs=args.baseline_epochs,
                            class_weights=class_weights, val_loader=val_loader)
        p = collect_probs(model, ctx, device, eval_cfg)
        meta = dict(name=name, kind="baseline", width=width, train_eps=np.nan, adv_ratio=np.nan)
        add_results(rows, curves, meta, p, targets)
        save(rows, curves)

    # ---- Hardened grid -------------------------------------------------------
    for width in args.widths:
        for train_eps in args.train_eps:
            for ratio in args.adv_ratios:
                name = f"eps={train_eps} r={ratio} w={width}"
                print(f"\n=== hardened {name} ===")
                torch.manual_seed(args.seed)
                model = NIDSClassifier(input_dim=len(FEATURE_NAMES),
                                       hidden_dims=hidden_dims_for(width)).to(device)
                cfg = AdvTrainingConfig(
                    epochs=args.adv_epochs, adv_ratio=ratio,
                    attack_config=AttackConfig(epsilon=train_eps, alpha=train_eps / 6, num_steps=7))
                model, _ = adversarial_train(
                    model, train_loader, val_loader, pre.scaler, ctx["mutable"], ctx["integer"],
                    ctx["nonneg"], device, cfg, class_weights=class_weights)
                p = collect_probs(model, ctx, device, eval_cfg)
                meta = dict(name=name, kind="hardened", width=width, train_eps=train_eps, adv_ratio=ratio)
                add_results(rows, curves, meta, p, targets)
                save(rows, curves)

    # ---- Ranking at the chosen false-positive rate --------------------------
    res = pd.DataFrame(rows)
    pick = res[res["operating_point"] == op_name(args.summary_fpr)].copy()
    pick["balance"] = 0.5 * (pick["clean_recall"] + pick["recall_after_attack"])
    pick = pick.sort_values("balance", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"\n=== Models ranked at {args.summary_fpr:.0%} false-positive rate "
          f"(balance = mean of clean recall and recall under attack) ===")
    print(pick[["name", "test_fpr", "clean_recall", "recall_after_attack",
                "corrected_esr", "balance"]].round(4).to_string(index=False))

    plot_curves(pd.DataFrame(curves), args.train_eps, args.adv_ratios, args.widths)


if __name__ == "__main__":
    main()
