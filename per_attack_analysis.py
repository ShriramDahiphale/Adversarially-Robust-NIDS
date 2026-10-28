"""
per_attack_analysis.py - Which attack types does each model catch or miss?

The pipeline trains on binary labels, so per-attack-type behaviour was never
measured. This script keeps the original CICIDS2017 label names (DoS Hulk,
PortScan, DDoS, Bot, Web Attack ...) alongside the binary labels and reports,
for the baseline and the hardened model, per attack type:

  * clean recall (no attack),
  * recall after the constrained PGD attack,
  * corrected ESR (among flows detected when clean, the fraction evaded).

Run robustness_checks.py FIRST (without --load) so baseline.pt and hardened.pt
exist, then run this script from the same folder with the SAME --seed and
--width (the seed fixes the data split and scaler):

    python per_attack_analysis.py --csv "D:\\DL Project\\data\\combined_sample.csv" --seed 0

Two operating points are reported: the default 0.5 threshold, and a threshold
chosen on validation benign flows to give --fpr false positives (default 5%).

Attack types with very few test flows (Heartbleed, Infiltration, SQL injection
in a 20% sample) give unreliable percentages, so n_test is always shown and
rare types are left out of the plot (--min-count).

Outputs: per_attack_results.csv and per_attack_recall.png
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from main import build_schema, FEATURE_NAMES
from data_module import NIDSPreprocessor, NIDSClassifier, split_validation
from attack_module import ConstrainedTabularAttack, AttackConfig
from robustness_checks import craft
from threshold_analysis import malicious_prob, threshold_for_fpr


def load_raw_labels(csv_path, schema):
    """Original (string) labels for exactly the rows NIDSPreprocessor keeps."""
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df[schema.feature_names + [schema.label_column]].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    labels = df[schema.label_column].astype(str).str.strip()
    # Some CICIDS releases contain a broken character in the web-attack names.
    return labels.str.replace("\ufffd", "-", regex=False).to_numpy(dtype=object)


def per_type_rows(model_name, op_name, thr, p, labels_mal, rows):
    """One row per attack type plus an overall row, at one decision threshold."""
    fpr = float((p["ben_test"] >= thr).mean())
    detected = p["mal_clean"] >= thr
    evaded = p["mal_adv"] < thr
    groups = [("ALL MALICIOUS", np.ones(len(labels_mal), dtype=bool))]
    groups += [(t, labels_mal == t) for t in sorted(set(labels_mal))]
    for name, mask in groups:
        n = int(mask.sum())
        n_det = int(detected[mask].sum())
        rows.append(dict(
            model=model_name, operating_point=op_name, threshold=thr, test_fpr=fpr,
            attack_type=name, n_test=n,
            clean_recall=float(detected[mask].mean()),
            recall_after_attack=float((p["mal_adv"][mask] >= thr).mean()),
            corrected_esr=float((evaded[mask] & detected[mask]).sum() / n_det) if n_det else np.nan,
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=128,
                        help="Must match the width the models were trained with")
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--fpr", type=float, default=0.05, help="Target false-positive rate, second operating point")
    parser.add_argument("--min-count", type=int, default=20, help="Minimum test flows for a type to be plotted")
    parser.add_argument("--baseline-path", default="baseline.pt")
    parser.add_argument("--hardened-path", default="hardened.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Same preprocessing and split as the other scripts -------------------
    schema = build_schema()
    pre = NIDSPreprocessor(schema, random_state=args.seed)
    df = pre.load_and_clean(args.csv)
    X_train, X_test, y_train, y_test = pre.fit_transform_split(df)
    X_tr, X_val, y_tr, y_val = split_validation(X_train, y_train, val_frac=0.1, seed=args.seed)

    # ---- Original attack-type names for the test flows -----------------------
    raw_labels = load_raw_labels(args.csv, schema)
    y_all = df[schema.label_column].values
    assert len(raw_labels) == len(y_all), "Row count mismatch between raw labels and cleaned data"
    binary_from_raw = np.array([0 if str(s).strip().upper() == "BENIGN" else 1 for s in raw_labels])
    assert np.array_equal(binary_from_raw, y_all), "Raw labels do not match the binary labels"
    # Same split call as NIDSPreprocessor.fit_transform_split, applied to row indices.
    _, idx_test = train_test_split(np.arange(len(y_all)), test_size=pre.test_size,
                                   random_state=args.seed, stratify=y_all)
    assert np.array_equal(y_all[idx_test], y_test), "Test split does not line up with X_test"
    labels_test = raw_labels[idx_test]
    labels_mal = labels_test[y_test == 1]

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
    hidden = (args.width, max(args.width // 2, 8), max(args.width // 4, 8))

    # ---- Per model: probabilities, then per-type tables ----------------------
    rows = []
    for name, path in [("baseline", args.baseline_path), ("hardened", args.hardened_path)]:
        model = NIDSClassifier(input_dim=len(FEATURE_NAMES), hidden_dims=hidden).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"[per-attack] Loaded {name} from {path}; crafting adversarial flows ...")
        attacker = ConstrainedTabularAttack(model, pre.scaler, mutable_mask, integer_mask,
                                            non_negative_mask, device)
        X_adv = craft(attacker, X_mal, y_mal, attack_cfg)
        p = dict(
            ben_val=malicious_prob(model, X_val_ben, device),
            ben_test=malicious_prob(model, X_ben, device),
            mal_clean=malicious_prob(model, X_mal, device),
            mal_adv=malicious_prob(model, X_adv, device),
        )
        per_type_rows(name, "default 0.5", 0.5, p, labels_mal, rows)
        per_type_rows(name, f"target FPR {args.fpr:.1%}", threshold_for_fpr(p["ben_val"], args.fpr),
                      p, labels_mal, rows)

    res = pd.DataFrame(rows)
    res.to_csv("per_attack_results.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    wides = {}
    for op in res["operating_point"].unique():
        sub = res[res["operating_point"] == op]
        fprs = sub.groupby("model")["test_fpr"].first()
        wide = sub.pivot(index="attack_type", columns="model",
                         values=["clean_recall", "recall_after_attack"])
        wide.columns = [f"{m}_{'clean' if a == 'clean_recall' else 'attacked'}" for a, m in wide.columns]
        wide.insert(0, "n_test", sub[sub["model"] == "baseline"].set_index("attack_type")["n_test"])
        wide = wide.sort_values("n_test", ascending=False)
        wides[op] = wide
        print(f"\n=== Recall by attack type, {op} (test FPR: baseline {fprs['baseline']:.2%}, "
              f"hardened {fprs['hardened']:.2%}; epsilon {args.epsilon}) ===")
        print(wide.round(4).to_string())

    # ---- Plot (default threshold, types with enough test flows) --------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    op = "default 0.5"
    w = wides[op]
    w = w[(w.index != "ALL MALICIOUS") & (w["n_test"] >= args.min_count)].sort_values("n_test")
    y = np.arange(len(w))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, 0.5 * len(w) + 2)), sharey=True)
    for ax, kind, title in [(axes[0], "clean", "Clean recall"), (axes[1], "attacked", "Recall under attack")]:
        ax.barh(y - 0.2, w[f"baseline_{kind}"], height=0.4, label="baseline", color="tab:blue")
        ax.barh(y + 0.2, w[f"hardened_{kind}"], height=0.4, label="hardened", color="tab:orange")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Recall")
        ax.set_title(f"{title} (epsilon {args.epsilon}, default threshold)" if kind == "attacked" else title)
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(loc="lower right")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{t} (n={int(n)})" for t, n in zip(w.index, w["n_test"])])
    plt.tight_layout()
    plt.savefig("per_attack_recall.png", dpi=150)
    print("\nSaved per_attack_results.csv and per_attack_recall.png")


if __name__ == "__main__":
    main()
