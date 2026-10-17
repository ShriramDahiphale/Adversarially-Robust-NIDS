"""
experiments.py (v2) - Multi-seed runs and epsilon sweep for the robust NIDS.

Put this file next to main.py and robustness_checks.py, then run:

    python experiments.py --csv "D:\\DL Project\\data\\combined_sample.csv"

Quick smoke test first (a few minutes):

    python experiments.py --csv "D:\\DL Project\\data\\combined_sample.csv" \
        --seeds 0 --baseline-epochs 2 --adv-epochs 4 --epsilons 0.1 0.3

For each seed it:
  1. makes a new train / validation / test split and trains a baseline and a
     class-weighted adversarially trained model (best epoch chosen on the
     validation set; training-time epsilon stays at 0.3),
  2. attacks both models on the TEST set at every epsilon in --epsilons
     (40-step constrained PGD),
  3. records clean recall, recall after attack, corrected ESR (only flows the
     model detected when clean) and the false-positive rate.

Outputs (in the current folder):
  sweep_results.csv  - one row per seed x model x epsilon
  epsilon_sweep.png  - corrected ESR and recall after attack vs epsilon,
                       mean +/- std over seeds
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
from robustness_checks import predict, craft


def run_seed(seed, df, schema, args, device, rows):
    """Train both models for one seed and evaluate them across all epsilons."""
    print(f"\n################ SEED {seed} ################")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # A different split per seed, so variance includes the data split too.
    pre = NIDSPreprocessor(schema, random_state=seed)
    X_train, X_test, y_train, y_test = pre.fit_transform_split(df)
    X_tr, X_val, y_tr, y_val = split_validation(X_train, y_train, val_frac=0.1, seed=seed)
    train_loader, test_loader = make_dataloaders(X_tr, y_tr, X_test, y_test, args.batch_size)
    val_loader = make_eval_loader(X_val, y_val)

    mutable_mask = schema.mutable_mask()
    integer_mask = schema.integer_mask()
    non_negative_mask = schema.non_negative_mask()
    class_weights = compute_class_weights(y_tr)

    print("\n=== Training baseline ===")
    hidden = (args.width, max(args.width // 2, 8), max(args.width // 4, 8))
    baseline = NIDSClassifier(input_dim=len(FEATURE_NAMES), hidden_dims=hidden)
    baseline = train_model(baseline, train_loader, device, epochs=args.baseline_epochs,
                           class_weights=class_weights, val_loader=val_loader)

    print("\n=== Adversarial training ===")
    hardened = NIDSClassifier(input_dim=len(FEATURE_NAMES), hidden_dims=hidden).to(device)
    cfg = AdvTrainingConfig(epochs=args.adv_epochs, adv_ratio=0.5)
    hardened, _ = adversarial_train(hardened, train_loader, val_loader, pre.scaler,
                                    mutable_mask, integer_mask, non_negative_mask,
                                    device, cfg, class_weights=class_weights)

    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    X_mal, y_mal = X_test_t[y_test_t == 1], y_test_t[y_test_t == 1]
    X_ben = X_test_t[y_test_t == 0]

    for name, model in [("baseline", baseline), ("hardened", hardened)]:
        model.eval()
        attacker = ConstrainedTabularAttack(model, pre.scaler, mutable_mask, integer_mask,
                                            non_negative_mask, device)

        detected = predict(model, X_mal, device) == 1
        n_det = max(int(detected.sum()), 1)
        clean_recall = detected.float().mean().item()
        fpr = (predict(model, X_ben, device) == 1).float().mean().item()

        # Epsilon balls are nested: a flow evaded at a small epsilon is also evadable at
        # every larger one. So the true worst case can only grow with epsilon, and
        # evaded_union (evaded at this epsilon OR any smaller one in the sweep) is a
        # valid, never-decreasing lower bound on it.
        evaded_union = torch.zeros(len(X_mal), dtype=torch.bool)
        for eps in sorted(args.epsilons):
            attack_cfg = AttackConfig(epsilon=eps, alpha=eps / 6, num_steps=40)
            X_adv = craft(attacker, X_mal, y_mal, attack_cfg)
            adv_pred = predict(model, X_adv, device)

            corrected_esr = ((adv_pred == 0) & detected).sum().item() / n_det
            recall_after = (adv_pred == 1).float().mean().item()
            evaded_union |= (adv_pred == 0)
            best_esr = ((evaded_union) & detected).sum().item() / n_det
            rows.append(dict(seed=seed, model=name, epsilon=eps, clean_recall=clean_recall,
                             recall_after=recall_after, corrected_esr=corrected_esr,
                             corrected_esr_best=best_esr, fpr=fpr))
            print(f"[seed {seed}] {name:8s} eps={eps:<5} clean recall={clean_recall:.2%} "
                  f"recall after attack={recall_after:.2%} corrected ESR={corrected_esr:.2%} "
                  f"best-of-sweep ESR={best_esr:.2%} FPR={fpr:.2%}")

    # Save after every seed so a crash does not lose finished work.
    pd.DataFrame(rows).to_csv(f"sweep_results_{args.tag}.csv", index=False)


def summarize_and_plot(rows, tag):
    res = pd.DataFrame(rows)
    metrics = ["clean_recall", "recall_after", "corrected_esr", "corrected_esr_best", "fpr"]
    summary = res.groupby(["model", "epsilon"])[metrics].agg(["mean", "std"])

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Summary (mean and std over seeds) ===")
    print(summary.round(4))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [("corrected_esr", "Corrected evasion success rate"),
              ("corrected_esr_best", "Best-of-sweep corrected ESR"),
              ("recall_after", "Recall after attack")]
    for ax, (metric, title) in zip(axes, panels):
        for model in ["baseline", "hardened"]:
            sub = summary.loc[model]
            ax.errorbar(sub.index, sub[(metric, "mean")], yerr=sub[(metric, "std")].fillna(0),
                        marker="o", capsize=3, label=model)
        ax.set_xlabel("Attack epsilon (standardized units)")
        ax.set_ylabel(title)
        ax.set_title(title + " vs epsilon")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"epsilon_sweep_{tag}.png", dpi=150)
    print(f"\nSaved sweep_results_{tag}.csv and epsilon_sweep_{tag}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.05, 0.1, 0.3, 0.5])
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--adv-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=128,
                        help="First hidden-layer width; layers are (w, w/2, w/4). 128 = original model")
    parser.add_argument("--tag", default=None,
                        help="Suffix for output files (default: w<width>), so runs never overwrite each other")
    args = parser.parse_args()
    args.tag = args.tag or f"w{args.width}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[experiments] Using device: {device}")

    schema = build_schema()
    df = NIDSPreprocessor(schema).load_and_clean(args.csv)  # clean once, reuse for every seed

    rows = []
    for seed in args.seeds:
        run_seed(seed, df, schema, args, device, rows)

    summarize_and_plot(rows, args.tag)


if __name__ == "__main__":
    main()
