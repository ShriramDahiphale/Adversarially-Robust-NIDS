"""
robustness_checks.py (v2) - Sanity checks for the adversarially trained NIDS.

Put this file in the same folder as main.py and run:

    python robustness_checks.py --csv "D:\\DL Project\\data\\combined_sample.csv" \
        --baseline-epochs 20 --adv-epochs 15 --seed 0

What it checks, and why
------------------------
1. Corrected ESR: the fraction of malicious flows the model DETECTED when clean
   that become evaded after the attack (removes flows already missed).
2. Clean recall vs adversarial recall: if recall is HIGHER after the attack,
   the model may be detecting perturbation artifacts instead of being robust.
3. Transfer attack: adversarial flows crafted against the BASELINE are fed to
   the hardened model.
4. Strong attack: 100 PGD steps with several random restarts against the
   hardened model; a flow counts as evaded if ANY restart evades.
5. False-positive rate on clean benign flows (the cost of hardening).

v2: models are selected on a validation split (never the test set). Models are
saved to baseline.pt / hardened.pt; re-run with --load to skip training.
Old .pt files from v1 are not comparable, so run once WITHOUT --load first.
"""

import argparse
import os

import numpy as np
import torch

from main import build_schema, FEATURE_NAMES
from data_module import (
    NIDSPreprocessor, make_dataloaders, make_eval_loader, split_validation,
    NIDSClassifier, train_model, compute_class_weights,
)
from attack_module import ConstrainedTabularAttack, AttackConfig
from defense_module import AdvTrainingConfig, adversarial_train


@torch.no_grad()
def predict(model, X, device, bs=2048):
    """Hard class predictions (0/1) for a tensor of standardized flows."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(device)).argmax(dim=1).cpu())
    return torch.cat(out)


def craft(attacker, X, y, cfg, bs=256):
    """Run the constrained attack in batches; returns adversarial tensor on CPU."""
    chunks = [attacker.generate(X[i:i + bs], y[i:i + bs], cfg).cpu()
              for i in range(0, len(X), bs)]
    return torch.cat(chunks, dim=0)


def corrected_report(name, model, X_mal, X_adv, device):
    """Print clean recall, adversarial recall, plain ESR and corrected ESR."""
    clean_pred = predict(model, X_mal, device)
    adv_pred = predict(model, X_adv, device)

    detected = clean_pred == 1
    n_det = int(detected.sum())
    clean_recall = detected.float().mean().item()
    adv_recall = (adv_pred == 1).float().mean().item()
    plain_esr = (adv_pred == 0).float().mean().item()
    corrected_esr = ((adv_pred == 0) & detected).sum().item() / max(n_det, 1)

    print(f"\n--- {name} ---")
    print(f"  Clean recall (malicious detected, no attack): {clean_recall:.2%}")
    print(f"  Recall after attack:                          {adv_recall:.2%}")
    print(f"  Plain ESR (all malicious flows):              {plain_esr:.2%}")
    print(f"  Corrected ESR (only originally-detected):     {corrected_esr:.2%}  "
          f"({int(((adv_pred == 0) & detected).sum())}/{n_det})")
    if adv_recall > clean_recall:
        print("  WARNING: recall is HIGHER after the attack than before it. The model "
              "may be detecting perturbation artifacts rather than being robust.")
    return corrected_esr


def strong_attack_report(model, attacker, X_mal, y_mal, eps, steps, restarts, device):
    """Worst case over several random-restart PGD runs against the model."""
    clean_pred = predict(model, X_mal, device)
    detected = clean_pred == 1
    n_det = int(detected.sum())

    evaded_any = torch.zeros(len(X_mal), dtype=torch.bool)
    cfg = AttackConfig(epsilon=eps, alpha=eps / 20, num_steps=steps, random_start=True)
    for r in range(restarts):
        X_adv = craft(attacker, X_mal, y_mal, cfg)
        evaded_any |= (predict(model, X_adv, device) == 0)
        so_far = ((evaded_any) & detected).sum().item() / max(n_det, 1)
        print(f"  restart {r + 1}/{restarts}: corrected ESR so far = {so_far:.2%}")

    final = ((evaded_any) & detected).sum().item() / max(n_det, 1)
    print(f"\n--- Hardened model, strong attack ({steps} steps x {restarts} restarts) ---")
    print(f"  Corrected ESR (worst case over restarts): {final:.2%}")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--adv-epochs", type=int, default=15)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--strong-steps", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load", action="store_true",
                        help="Load baseline.pt / hardened.pt instead of retraining")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[checks] Using device: {device}")

    # ---- Data: train / validation / test ------------------------------------
    schema = build_schema()
    pre = NIDSPreprocessor(schema, random_state=args.seed)
    df = pre.load_and_clean(args.csv)
    X_train, X_test, y_train, y_test = pre.fit_transform_split(df)
    X_tr, X_val, y_tr, y_val = split_validation(X_train, y_train, val_frac=0.1, seed=args.seed)
    train_loader, test_loader = make_dataloaders(X_tr, y_tr, X_test, y_test, args.batch_size)
    val_loader = make_eval_loader(X_val, y_val)
    class_weights = compute_class_weights(y_tr)

    mutable_mask = schema.mutable_mask()
    integer_mask = schema.integer_mask()
    non_negative_mask = schema.non_negative_mask()

    # ---- Models: train or load ----------------------------------------------
    baseline = NIDSClassifier(input_dim=len(FEATURE_NAMES)).to(device)
    hardened = NIDSClassifier(input_dim=len(FEATURE_NAMES)).to(device)

    if args.load and os.path.exists("baseline.pt") and os.path.exists("hardened.pt"):
        baseline.load_state_dict(torch.load("baseline.pt", map_location=device))
        hardened.load_state_dict(torch.load("hardened.pt", map_location=device))
        print("[checks] Loaded baseline.pt and hardened.pt")
    else:
        print("\n=== Training baseline ===")
        baseline = train_model(baseline, train_loader, device, epochs=args.baseline_epochs,
                               class_weights=class_weights, val_loader=val_loader)
        print("\n=== Adversarial training ===")
        cfg = AdvTrainingConfig(epochs=args.adv_epochs, adv_ratio=0.5)
        hardened, _ = adversarial_train(hardened, train_loader, val_loader, pre.scaler,
                                        mutable_mask, integer_mask, non_negative_mask,
                                        device, cfg, class_weights=class_weights)
        torch.save(baseline.state_dict(), "baseline.pt")
        torch.save(hardened.state_dict(), "hardened.pt")
        print("[checks] Saved baseline.pt and hardened.pt")

    baseline.eval()
    hardened.eval()

    # ---- Test sets (test data is only used from here on) --------------------
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    X_mal, y_mal = X_test_t[y_test_t == 1], y_test_t[y_test_t == 1]
    X_ben = X_test_t[y_test_t == 0]

    atk_base = ConstrainedTabularAttack(baseline, pre.scaler, mutable_mask, integer_mask,
                                        non_negative_mask, device)
    atk_hard = ConstrainedTabularAttack(hardened, pre.scaler, mutable_mask, integer_mask,
                                        non_negative_mask, device)
    std_cfg = AttackConfig(epsilon=args.epsilon, alpha=args.epsilon / 6, num_steps=40)

    print("\n=== Crafting adversarial malicious flows (standard 40-step PGD) ===")
    adv_on_base = craft(atk_base, X_mal, y_mal, std_cfg)
    adv_on_hard = craft(atk_hard, X_mal, y_mal, std_cfg)

    esr_base = corrected_report("Baseline, white-box attack", baseline, X_mal, adv_on_base, device)
    esr_hard = corrected_report("Hardened, white-box attack", hardened, X_mal, adv_on_hard, device)
    esr_transfer = corrected_report("Hardened, TRANSFER attack (crafted on baseline)",
                                    hardened, X_mal, adv_on_base, device)

    print("\n=== Strong attack on hardened model ===")
    esr_strong = strong_attack_report(hardened, atk_hard, X_mal, y_mal, args.epsilon,
                                      args.strong_steps, args.restarts, device)

    fpr_base = (predict(baseline, X_ben, device) == 1).float().mean().item()
    fpr_hard = (predict(hardened, X_ben, device) == 1).float().mean().item()

    print("\n=== Summary (corrected ESR = only flows detected when clean) ===")
    print(f"  Baseline  white-box:           {esr_base:.2%}")
    print(f"  Hardened  white-box (40 step): {esr_hard:.2%}")
    print(f"  Hardened  transfer attack:     {esr_transfer:.2%}")
    print(f"  Hardened  strong attack:       {esr_strong:.2%}")
    print(f"  False-positive rate  baseline: {fpr_base:.2%}   hardened: {fpr_hard:.2%}")


if __name__ == "__main__":
    main()
