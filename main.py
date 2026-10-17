"""
main.py (v2) - End-to-end pipeline for the Adversarially Robust NIDS project.

Stages:
  1. Load & preprocess the CICIDS-style CSV; split train / validation / test.
  2. Train the baseline classifier (LR decay, best epoch chosen on validation).
  3. Attack the baseline with constrained PGD and report Evasion Success Rate.
  4. Adversarially train a hardened model (warmup, epsilon ramp, LR decay,
     best epoch chosen on validation clean + adversarial score).
  5. Attack the hardened model with the SAME PGD engine on the TEST set and
     report the new Evasion Success Rate and the trade-off plot.

Usage:
    python main.py --csv "D:\\DL Project\\data\\combined_sample.csv" --seed 0

The test set is only used in stages 3 and 5; all model selection uses the
validation set.
"""

import argparse

import numpy as np
import torch

from data_module import (
    FeatureSchema, NIDSPreprocessor, make_dataloaders, make_eval_loader, split_validation,
    NIDSClassifier, train_model, evaluate_model, compute_class_weights,
    CICIDS_DEFAULT_IMMUTABLE, CICIDS_DEFAULT_INTEGER, CICIDS_DEFAULT_NONNEG,
)
from attack_module import (
    ConstrainedTabularAttack, AttackConfig, evasion_success_rate,
    craft_adversarial_malicious_set,
)
from defense_module import AdvTrainingConfig, adversarial_train, evaluate_hardened_model


# Feature set for the CICIDS2017 "MachineLearningCSV" release (no Protocol column).
FEATURE_NAMES = [
    "Destination Port", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Bytes/s", "Flow Packets/s",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
]
LABEL_COLUMN = "Label"


def build_schema() -> FeatureSchema:
    return FeatureSchema(
        feature_names=FEATURE_NAMES,
        label_column=LABEL_COLUMN,
        immutable_features=[f for f in CICIDS_DEFAULT_IMMUTABLE if f in FEATURE_NAMES],
        integer_features=[f for f in CICIDS_DEFAULT_INTEGER if f in FEATURE_NAMES],
        non_negative_features=[f for f in CICIDS_DEFAULT_NONNEG if f in FEATURE_NAMES],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CICIDS-style CSV file")
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--adv-epochs", type=int, default=15)
    parser.add_argument("--epsilon", type=float, default=0.3, help="L_inf budget, standardized units")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Using device: {device}")

    # ---- Stage 1: Preprocess and split --------------------------------------
    schema = build_schema()
    pre = NIDSPreprocessor(schema, random_state=args.seed)
    df = pre.load_and_clean(args.csv)
    X_train, X_test, y_train, y_test = pre.fit_transform_split(df)
    X_tr, X_val, y_tr, y_val = split_validation(X_train, y_train, val_frac=0.1, seed=args.seed)
    train_loader, test_loader = make_dataloaders(X_tr, y_tr, X_test, y_test, args.batch_size)
    val_loader = make_eval_loader(X_val, y_val)
    class_weights = compute_class_weights(y_tr)

    # ---- Stage 2: Baseline classifier ---------------------------------------
    print("\n=== Stage 2: Training baseline NIDS classifier ===")
    baseline = NIDSClassifier(input_dim=len(FEATURE_NAMES))
    baseline = train_model(baseline, train_loader, device, epochs=args.baseline_epochs,
                           class_weights=class_weights, val_loader=val_loader)
    evaluate_model(baseline, test_loader, device, tag="Baseline Clean Test")

    # ---- Stage 3: Attack the baseline ---------------------------------------
    print("\n=== Stage 3: Attacking baseline with constrained PGD ===")
    mutable_mask = schema.mutable_mask()
    integer_mask = schema.integer_mask()
    non_negative_mask = schema.non_negative_mask()

    attacker = ConstrainedTabularAttack(baseline, pre.scaler, mutable_mask, integer_mask,
                                        non_negative_mask, device)
    attack_cfg = AttackConfig(epsilon=args.epsilon, alpha=args.epsilon / 6, num_steps=40)
    X_adv, y_mal, _ = craft_adversarial_malicious_set(attacker, X_test, y_test, attack_cfg, device)
    baseline_esr = evasion_success_rate(baseline, X_adv, y_mal, device)

    # ---- Stage 4: Adversarial training (hardening) --------------------------
    print("\n=== Stage 4: Adversarial training (defensive hardening) ===")
    hardened = NIDSClassifier(input_dim=len(FEATURE_NAMES))
    adv_train_cfg = AdvTrainingConfig(epochs=args.adv_epochs, adv_ratio=0.5)
    hardened, tradeoff_logger = adversarial_train(
        hardened, train_loader, val_loader, pre.scaler,
        mutable_mask, integer_mask, non_negative_mask, device, adv_train_cfg,
        class_weights=class_weights
    )

    # ---- Stage 5: Attack the hardened model on the test set -----------------
    print("\n=== Stage 5: Attacking hardened model (final comparison) ===")
    evaluate_model(hardened, test_loader, device, tag="Hardened Clean Test")
    hardened_metrics = evaluate_hardened_model(
        hardened, test_loader, pre.scaler, mutable_mask, integer_mask,
        non_negative_mask, device, attack_cfg
    )

    print("\n=== Summary ===")
    print(f"Baseline model  - Evasion Success Rate: {baseline_esr['evasion_success_rate']:.2%}")
    print(f"Hardened model  - Evasion Success Rate: {hardened_metrics['evasion_success_rate']:.2%}")
    print("(Plain ESR counts flows missed even without attack; see robustness_checks.py "
          "for the corrected ESR.)")
    tradeoff_logger.plot()


if __name__ == "__main__":
    main()
