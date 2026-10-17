"""
threshold_analysis.py - Choose an operating point for the NIDS.

The hardened model flags too many benign flows at the default 0.5 decision
threshold. Raising the threshold trades some recall for fewer false alarms,
without retraining. This script shows that trade-off, clean AND under attack.

Run robustness_checks.py FIRST (without --load). It trains both models and
saves baseline.pt and hardened.pt. Then run this script from the same folder,
with the SAME --seed (the seed fixes the train/validation/test split, and the
scaler must match the one the models were trained with):

    python threshold_analysis.py --csv "D:\\DL Project\\data\\combined_sample.csv" --seed 0

How it works
-------------
* Thresholds are picked on VALIDATION benign flows: for a target false-positive
  rate f, the threshold is the (1 - f) quantile of the model's malicious
  probability on validation benign flows. The test set is not used to choose
  thresholds.
* Adversarial malicious test flows are crafted once per model (40-step
  constrained PGD). PGD maximises the loss on the true label, which drives the
  malicious probability down, so the same adversarial flows are reused at every
  threshold. A threshold-aware attacker could do somewhat better, so treat the
  numbers at high thresholds as slightly optimistic.
* For each model and threshold it reports test false-positive rate, clean
  recall, recall after attack, and corrected ESR (only flows detected when
  clean that become evaded).

Outputs: threshold_results.csv and threshold_tradeoff.png
"""

import argparse

import numpy as np
import pandas as pd
import torch

from main import build_schema, FEATURE_NAMES
from data_module import NIDSPreprocessor, NIDSClassifier, split_validation
from attack_module import ConstrainedTabularAttack, AttackConfig
from robustness_checks import craft


@torch.no_grad()
def malicious_prob(model, X, device, bs=4096):
    """Probability of the malicious class for each flow."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        logits = model(X[i:i + bs].to(device))
        out.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(out).numpy()


def metrics_at(thr, p_ben_test, p_mal_clean, p_mal_adv):
    """False-positive rate, clean recall, recall after attack, corrected ESR."""
    fpr = float((p_ben_test >= thr).mean())
    detected = p_mal_clean >= thr
    clean_recall = float(detected.mean())
    recall_after = float((p_mal_adv >= thr).mean())
    n_det = max(int(detected.sum()), 1)
    corrected_esr = float(((p_mal_adv < thr) & detected).sum() / n_det)
    return fpr, clean_recall, recall_after, corrected_esr


def threshold_for_fpr(p_ben_val, target_fpr):
    """Threshold giving about `target_fpr` false positives on validation benign flows."""
    return float(np.quantile(p_ben_val, 1.0 - target_fpr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--baseline-path", default="baseline.pt")
    parser.add_argument("--hardened-path", default="hardened.pt")
    parser.add_argument("--targets", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05, 0.10],
                        help="Target false-positive rates for the table")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Same preprocessing and split as robustness_checks.py ---------------
    schema = build_schema()
    pre = NIDSPreprocessor(schema, random_state=args.seed)
    df = pre.load_and_clean(args.csv)
    X_train, X_test, y_train, y_test = pre.fit_transform_split(df)
    X_tr, X_val, y_tr, y_val = split_validation(X_train, y_train, val_frac=0.1, seed=args.seed)

    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    X_val_ben = X_val_t[y_val_t == 0]
    X_ben = X_test_t[y_test_t == 0]
    X_mal, y_mal = X_test_t[y_test_t == 1], y_test_t[y_test_t == 1]

    mutable_mask = schema.mutable_mask()
    integer_mask = schema.integer_mask()
    non_negative_mask = schema.non_negative_mask()
    attack_cfg = AttackConfig(epsilon=args.epsilon, alpha=args.epsilon / 6, num_steps=40)

    # ---- Load models and compute probabilities ------------------------------
    probs = {}
    for name, path in [("baseline", args.baseline_path), ("hardened", args.hardened_path)]:
        model = NIDSClassifier(input_dim=len(FEATURE_NAMES)).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"[threshold] Loaded {name} from {path}; crafting adversarial malicious flows ...")
        attacker = ConstrainedTabularAttack(model, pre.scaler, mutable_mask, integer_mask,
                                            non_negative_mask, device)
        X_adv = craft(attacker, X_mal, y_mal, attack_cfg)
        probs[name] = dict(
            ben_val=malicious_prob(model, X_val_ben, device),
            ben_test=malicious_prob(model, X_ben, device),
            mal_clean=malicious_prob(model, X_mal, device),
            mal_adv=malicious_prob(model, X_adv, device),
        )

    # ---- Table at chosen operating points -----------------------------------
    rows = []
    for name, p in probs.items():
        points = [("default 0.5", 0.5)] + [(f"target FPR {t:.0%}", threshold_for_fpr(p["ben_val"], t))
                                           for t in args.targets]
        for label, thr in points:
            fpr, rec, rec_adv, esr = metrics_at(thr, p["ben_test"], p["mal_clean"], p["mal_adv"])
            rows.append(dict(model=name, operating_point=label, threshold=thr, test_fpr=fpr,
                             clean_recall=rec, recall_after_attack=rec_adv, corrected_esr=esr))

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"\n=== Operating points (epsilon = {args.epsilon}, 40-step PGD, test set) ===")
    print(table.round(4).to_string(index=False))
    table.to_csv("threshold_results.csv", index=False)

    # ---- Plot: recall and corrected ESR vs false-positive rate ---------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.logspace(-3, np.log10(0.3), 40)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"baseline": "tab:blue", "hardened": "tab:orange"}
    for name, p in probs.items():
        fprs, recs, recs_adv, esrs = [], [], [], []
        for t in grid:
            thr = threshold_for_fpr(p["ben_val"], t)
            fpr, rec, rec_adv, esr = metrics_at(thr, p["ben_test"], p["mal_clean"], p["mal_adv"])
            fprs.append(fpr)
            recs.append(rec)
            recs_adv.append(rec_adv)
            esrs.append(esr)
        axes[0].plot(fprs, recs, color=colors[name], label=f"{name}, clean")
        axes[0].plot(fprs, recs_adv, color=colors[name], linestyle="--", label=f"{name}, under attack")
        axes[1].plot(fprs, esrs, color=colors[name], label=name)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Test false-positive rate")
    axes[0].set_ylabel("Recall on malicious flows")
    axes[0].set_title("Recall vs false-positive rate")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Test false-positive rate")
    axes[1].set_ylabel("Corrected evasion success rate")
    axes[1].set_title("Corrected ESR vs false-positive rate")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("threshold_tradeoff.png", dpi=150)
    print("\nSaved threshold_results.csv and threshold_tradeoff.png")


if __name__ == "__main__":
    main()
