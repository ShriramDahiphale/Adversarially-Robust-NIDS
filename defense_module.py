"""
Module 3 (v2): Adversarial Training & Defensive Hardening
===========================================================

Changes in v2 (all aimed at the instability seen across seeds)
----------------------------------------------------------------
1. Clean warmup: the first epochs train on clean data only. Adversarial
   examples crafted against a randomly initialised model are meaningless and
   make early training erratic.
2. Epsilon ramp: after warmup the attack budget grows linearly from a small
   value to the full epsilon (a simple curriculum).
3. Cosine learning-rate decay and gradient clipping.
4. Validation-based checkpoint selection: every epoch the model is scored on a
   held-out VALIDATION set, both clean and under a (cheaper) PGD attack, and
   the best epoch is restored at the end. The score is the mean of clean F1 and
   adversarial F1, so the chosen checkpoint balances accuracy and robustness.
   The test set is never used for selection.
5. The trade-off log now holds clean AND adversarial metrics for every epoch,
   measured on the same validation set, so the plot is a fair comparison.

Security/ML rationale (unchanged)
----------------------------------
Adversarial training (Goodfellow et al. 2015; Madry et al. 2018) crafts
adversarial examples from the model being trained and mixes them into the
batches, using the SAME domain constraints as the attack engine (an attacker
cannot cheat the constraints, so the defence need not cover infeasible
perturbations). Robustness usually costs some clean performance, so both are
logged to make the trade-off visible.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from attack_module import ConstrainedTabularAttack, AttackConfig, evasion_success_rate
from data_module import TabularFlowDataset, evaluate_model


@dataclass
class AdvTrainingConfig:
    epochs: int = 15
    lr: float = 1e-3
    adv_ratio: float = 0.5               # fraction of attacked samples swapped for adversarial versions
    attack_config: AttackConfig = None   # training-time PGD (target epsilon/alpha/steps)
    warmup_epochs: int = 2               # clean-only epochs before adversarial examples start
    eps_ramp_epochs: int = 4             # epochs to ramp epsilon from small to full after warmup
    grad_clip: float = 5.0               # max gradient norm
    perturb_both_classes: bool = True    # True = also perturb benign flows, so "perturbed" cannot predict "malicious"
    val_attack_steps: int = 10           # PGD steps for the per-epoch validation attack
    val_max_malicious: int = 3000        # cap on malicious validation flows attacked each epoch
    val_max_benign: int = 12000          # cap on benign validation flows used each epoch

    def __post_init__(self):
        if self.attack_config is None:
            self.attack_config = AttackConfig(epsilon=0.3, alpha=0.05, num_steps=7)


class TradeoffLogger:
    """Collects (epoch, clean metrics, adversarial metrics) for the trade-off plot."""

    def __init__(self):
        self.records: List[dict] = []

    def log(self, epoch: int, clean: dict, adv: Optional[dict] = None):
        row = {"epoch": epoch, "clean_accuracy": clean["accuracy"], "clean_f1": clean["f1"]}
        if adv is not None:
            row["adv_accuracy"] = adv["accuracy"]
            row["adv_f1"] = adv["f1"]
        self.records.append(row)

    def print_table(self):
        print("\n[Trade-off Log, validation set] epoch | clean_acc | clean_f1 | adv_acc | adv_f1")
        for r in self.records:
            print(f"  {r['epoch']:>5} | {r['clean_accuracy']:.4f}   | {r['clean_f1']:.4f}   | "
                  f"{r.get('adv_accuracy', float('nan')):.4f}  | {r.get('adv_f1', float('nan')):.4f}")

    def plot(self, save_path: Optional[str] = None):
        """Plots clean vs adversarial accuracy and F1 per epoch (validation set)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [r["epoch"] for r in self.records]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, key, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "f1", "F1 score")]:
            ax.plot(epochs, [r[f"clean_{key}"] for r in self.records], marker="o", label="Clean")
            ax.plot(epochs, [r.get(f"adv_{key}", np.nan) for r in self.records], marker="s",
                    label="Under PGD attack")
            ax.set_xlabel("Adversarial training epoch")
            ax.set_ylabel(title)
            ax.set_title(f"{title}: clean vs adversarial (validation)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()

        out = save_path or "robustness_tradeoff.png"
        plt.savefig(out, dpi=150)
        print(f"[TradeoffLogger] Saved trade-off plot to {out}")
        return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@torch.no_grad()
def _predict(model: nn.Module, X: torch.Tensor, device: torch.device, bs: int = 2048):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(device)).argmax(dim=1).cpu())
    return torch.cat(out)


def _metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def _collect(loader: DataLoader):
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    return torch.cat(xs), torch.cat(ys)


def _prepare_validation(val_loader: DataLoader, max_malicious: int, max_benign: int):
    """Fixed random subsets of benign and malicious validation flows."""
    X, y = _collect(val_loader)
    X_ben, y_ben = X[y == 0], y[y == 0]
    X_mal, y_mal = X[y == 1], y[y == 1]
    g = torch.Generator().manual_seed(0)
    if len(X_mal) > max_malicious:
        keep = torch.randperm(len(X_mal), generator=g)[:max_malicious]
        X_mal, y_mal = X_mal[keep], y_mal[keep]
    if len(X_ben) > max_benign:
        keep = torch.randperm(len(X_ben), generator=g)[:max_benign]
        X_ben, y_ben = X_ben[keep], y_ben[keep]
    return X_ben, y_ben, X_mal, y_mal


def _attack_in_chunks(attacker, X, y, cfg: AttackConfig, bs: int = 512):
    chunks = [attacker.generate(X[i:i + bs], y[i:i + bs], cfg).cpu() for i in range(0, len(X), bs)]
    return torch.cat(chunks) if chunks else X


def _validation_scores(model, attacker, val_data, eval_cfg: AttackConfig, device,
                       attack_benign: bool = True):
    """
    Clean and adversarial metrics on the SAME validation flows, once as-is and
    once attacked. With attack_benign=True the benign flows are perturbed too.
    That matters: if only malicious flows were perturbed, a model that simply
    flags "perturbed-looking" inputs as malicious would score well, and
    checkpoint selection would reward that shortcut instead of real robustness.
    """
    X_ben, y_ben, X_mal, y_mal = val_data
    model.eval()
    X_mal_adv = _attack_in_chunks(attacker, X_mal, y_mal, eval_cfg)
    X_ben_adv = _attack_in_chunks(attacker, X_ben, y_ben, eval_cfg) if attack_benign else X_ben

    y_true = torch.cat([y_ben, y_mal]).numpy()
    clean = _metrics(y_true, torch.cat([_predict(model, X_ben, device),
                                        _predict(model, X_mal, device)]).numpy())
    adv = _metrics(y_true, torch.cat([_predict(model, X_ben_adv, device),
                                      _predict(model, X_mal_adv, device)]).numpy())
    return clean, adv


def _epoch_attack_cfg(base: AttackConfig, epoch: int, warmup: int, ramp: int) -> Optional[AttackConfig]:
    """None during clean warmup; afterwards epsilon/alpha ramp up to the full values."""
    if epoch <= warmup:
        return None
    scale = 1.0 if ramp <= 0 else min(1.0, (epoch - warmup) / ramp)
    return AttackConfig(epsilon=base.epsilon * scale, alpha=base.alpha * scale,
                        num_steps=base.num_steps, random_start=base.random_start)


# --------------------------------------------------------------------------
# Adversarial training loop
# --------------------------------------------------------------------------
def adversarial_train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                      scaler, mutable_mask: np.ndarray, integer_mask: np.ndarray,
                      non_negative_mask: np.ndarray, device: torch.device,
                      config: AdvTrainingConfig, eval_attack_config: Optional[AttackConfig] = None,
                      class_weights: Optional[torch.Tensor] = None):
    """
    Adversarial training with validation-based checkpoint selection.

    Each batch (after warmup): craft adversarial versions of the malicious
    samples (or of all samples if perturb_both_classes), swap `adv_ratio` of
    them into the batch, and train on the mixed batch. Each epoch: score the
    model on the validation set, clean and under attack, and remember the
    best checkpoint. The best checkpoint is restored before returning.

    NOTE: the third argument is the VALIDATION loader (not the test loader).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None)
    logger = TradeoffLogger()

    attacker = ConstrainedTabularAttack(
        model, scaler, mutable_mask, integer_mask, non_negative_mask, device)

    # Validation attack: full epsilon, fewer steps than the final evaluation.
    full_eps = config.attack_config.epsilon
    val_cfg = eval_attack_config or AttackConfig(
        epsilon=full_eps, alpha=full_eps / 4, num_steps=config.val_attack_steps)
    val_data = _prepare_validation(val_loader, config.val_max_malicious, config.val_max_benign)

    # Always leave at least one adversarial epoch, even for short smoke tests.
    warmup = min(config.warmup_epochs, max(config.epochs - 1, 0))

    best_score, best_state = -1.0, None
    for epoch in range(1, config.epochs + 1):
        model.train()
        ep_cfg = _epoch_attack_cfg(config.attack_config, epoch, warmup, config.eps_ramp_epochs)

        total_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            if ep_cfg is not None and config.adv_ratio > 0:
                if config.perturb_both_classes:
                    atk_mask = torch.ones_like(yb, dtype=torch.bool)
                else:
                    atk_mask = (yb == 1)
                if atk_mask.any():
                    # BatchNorm/Dropout must be in eval mode while crafting attacks.
                    model.eval()
                    X_adv = attacker.generate(xb[atk_mask], yb[atk_mask], ep_cfg)
                    model.train()

                    # idx = batch positions of attacked samples (ascending), so
                    # searchsorted maps each chosen position back to its row in
                    # X_adv while preserving the order of `swap`.
                    idx = torch.where(atk_mask)[0]
                    n_swap = int(round(config.adv_ratio * len(idx)))
                    swap = idx[torch.randperm(len(idx), device=idx.device)[:n_swap]]
                    pos = torch.searchsorted(idx, swap)
                    xb = xb.clone()
                    xb[swap] = X_adv[pos].to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Validation: clean + adversarial on the same flows.
        clean, adv = _validation_scores(model, attacker, val_data, val_cfg, device,
                                        attack_benign=config.perturb_both_classes)
        score = 0.5 * (clean["f1"] + adv["f1"])
        improved = score > best_score
        if improved:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        logger.log(epoch, clean, adv)

        phase = "clean warmup" if ep_cfg is None else f"eps {ep_cfg.epsilon:.2f}"
        print(f"[AdvTrain] Epoch {epoch:3d}/{config.epochs} | loss {total_loss / n_batches:.4f} | "
              f"lr {lr_now:.2e} | {phase} | val F1 clean {clean['f1']:.4f} adv {adv['f1']:.4f} | "
              f"score {score:.4f}{'  *best' if improved else ''}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[AdvTrain] Restored best epoch by validation score ({best_score:.4f})")

    logger.print_table()
    return model, logger


# --------------------------------------------------------------------------
# Final evaluation on the TEST set
# --------------------------------------------------------------------------
def evaluate_hardened_model(model: nn.Module, test_loader: DataLoader, scaler,
                            mutable_mask: np.ndarray, integer_mask: np.ndarray,
                            non_negative_mask: np.ndarray, device: torch.device,
                            attack_config: AttackConfig) -> dict:
    """
    Attack the hardened model on the full test set with the constrained PGD
    engine and report standard metrics plus the Evasion Success Rate. Metrics
    are computed on clean benign flows + adversarial malicious flows.
    """
    attacker = ConstrainedTabularAttack(model, scaler, mutable_mask, integer_mask, non_negative_mask, device)

    all_X, all_y = [], []
    for xb, yb in test_loader:
        all_X.append(xb)
        all_y.append(yb)
    X_all = torch.cat(all_X, dim=0)
    y_all = torch.cat(all_y, dim=0)

    mal_idx = (y_all == 1)
    X_mal, y_mal = X_all[mal_idx], y_all[mal_idx]

    batch_size = 256
    adv_chunks = []
    for i in range(0, len(X_mal), batch_size):
        adv_chunks.append(attacker.generate(X_mal[i:i + batch_size], y_mal[i:i + batch_size], attack_config).cpu())
    X_adv = torch.cat(adv_chunks, dim=0) if adv_chunks else X_mal

    esr = evasion_success_rate(model, X_adv, y_mal, device)

    X_benign = X_all[~mal_idx]
    y_benign = y_all[~mal_idx]
    X_mixed = torch.cat([X_benign, X_adv], dim=0)
    y_mixed = torch.cat([y_benign, y_mal], dim=0)

    mixed_loader = DataLoader(TabularFlowDataset(X_mixed.numpy(), y_mixed.numpy()), batch_size=256)
    metrics = evaluate_model(model, mixed_loader, device, tag="Adversarial (post-attack) Test Set")
    metrics.update(esr)
    return metrics
