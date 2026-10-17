"""
Module 2: Constraint-Aware Adversarial Attack Engine
======================================================

Security/ML rationale
----------------------
Standard PGD/FGSM (Madry et al., Goodfellow et al.) were designed for
image data, where "any point in the epsilon-ball" is a semantically
valid image. Tabular network traffic has no such freedom:

  * Some features are IMMUTABLE from the attacker's perspective. An
    attacker crafting an evasive attack flow cannot retroactively change
    the destination port they attacked, or the transport protocol,
    without the "attack" ceasing to be the attack they intended to send.
    -> we zero the gradient (freeze) on these dimensions every step.

  * Some features are INTEGER-VALUED counts (packet counts, flag counts).
    A continuous gradient step can produce "173.42 packets", which does
    not correspond to any flow that could exist on the wire.
    -> we round these dimensions back to the nearest integer after each
       step (in raw/unscaled units, then re-standardize).

  * Some features can never be NEGATIVE (durations, byte counts).
    -> we clamp these dimensions at 0 in raw units.

Because these constraints are only physically meaningful in the RAW
(pre-StandardScaler) feature space, this engine performs the perturbation
loop by moving back and forth between standardized space (where the model
operates and where gradients are computed) and raw space (where the
constraints are enforced). This is the key structural difference from a
textbook PGD implementation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AttackConfig:
    epsilon: float = 0.3          # Max L_inf perturbation, in STANDARDIZED units
    alpha: float = 0.05           # Step size per PGD iteration, in STANDARDIZED units
    num_steps: int = 20           # PGD iterations (num_steps=1 with alpha=epsilon ~= FGSM)
    random_start: bool = True     # Randomly initialize within the epsilon-ball (standard PGD practice)


class ConstrainedTabularAttack:
    """
    Constraint-aware PGD/FGSM attacker for tabular NIDS models.

    Parameters
    ----------
    model : the (frozen, eval-mode) classifier to attack.
    scaler : the fitted StandardScaler from Module 1's NIDSPreprocessor,
             needed to move between standardized <-> raw feature space.
    mutable_mask, integer_mask, non_negative_mask : np.ndarray (float32),
             per-feature masks aligned to the feature column order, as
             produced by FeatureSchema in Module 1.
    device : torch device.
    """

    def __init__(self, model: nn.Module, scaler, mutable_mask: np.ndarray,
                 integer_mask: np.ndarray, non_negative_mask: np.ndarray,
                 device: torch.device):
        self.model = model.to(device).eval()
        self.scaler = scaler
        self.device = device

        self.mutable_mask = torch.tensor(mutable_mask, dtype=torch.float32, device=device)
        self.integer_mask = torch.tensor(integer_mask, dtype=torch.float32, device=device)
        self.non_negative_mask = torch.tensor(non_negative_mask, dtype=torch.float32, device=device)

        # scaler.mean_ / scaler.scale_ let us convert standardized <-> raw
        # without round-tripping through pandas each step.
        self.scale_ = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
        self.mean_ = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)

    # -- helpers to move between standardized and raw feature space --------
    def _to_raw(self, x_std: torch.Tensor) -> torch.Tensor:
        return x_std * self.scale_ + self.mean_

    def _to_std(self, x_raw: torch.Tensor) -> torch.Tensor:
        return (x_raw - self.mean_) / self.scale_

    def _apply_domain_constraints(self, x_std: torch.Tensor) -> torch.Tensor:
        """
        Enforce integer-rounding and non-negativity in RAW space, then
        convert back to standardized space for the next gradient step.
        Mutability is enforced separately in the perturbation step itself
        (by masking the gradient / delta), not here.
        """
        x_raw = self._to_raw(x_std)

        # Round integer-count features to the nearest whole number.
        rounded = torch.round(x_raw)
        x_raw = self.integer_mask * rounded + (1 - self.integer_mask) * x_raw

        # Clamp non-negative features at 0.
        clamped = torch.clamp(x_raw, min=0.0)
        x_raw = self.non_negative_mask * clamped + (1 - self.non_negative_mask) * x_raw

        return self._to_std(x_raw)

    def generate(self, X: torch.Tensor, y_true: torch.Tensor, config: AttackConfig) -> torch.Tensor:
        """
        Runs constrained PGD (set config.num_steps=1 for FGSM-style single step).

        X : (batch, n_features) standardized inputs, typically MALICIOUS
            flows we want to perturb so the model misclassifies them as
            Benign (an "evasion" attack -- the realistic threat model for
            an attacker trying to sneak traffic past a NIDS).
        y_true : (batch,) ground-truth labels (all 1 = Malicious, expected).

        Returns adversarial examples X_adv, still in standardized space,
        satisfying:
          - |X_adv - X|_inf <= epsilon on mutable features
          - X_adv == X exactly on immutable features
          - integer features are whole numbers in raw space
          - non-negative features are >= 0 in raw space
        """
        X = X.clone().detach().to(self.device)
        y_true = y_true.to(self.device)
        criterion = nn.CrossEntropyLoss()

        if config.random_start:
            noise = torch.empty_like(X).uniform_(-config.epsilon, config.epsilon)
            noise = noise * self.mutable_mask  # never perturb immutable dims even at init
            X_adv = (X + noise).clone().detach()
            X_adv = self._apply_domain_constraints(X_adv)
        else:
            X_adv = X.clone().detach()

        for step in range(config.num_steps):
            X_adv.requires_grad_(True)
            logits = self.model(X_adv)
            # We ASCEND the loss on the true label to push the model away
            # from the correct (Malicious) prediction -- this is an
            # untargeted evasion attack from the attacker's point of view.
            loss = criterion(logits, y_true)
            grad = torch.autograd.grad(loss, X_adv)[0]

            # --- Core constraint #1: freeze immutable features ---
            # Zeroing the gradient on immutable dimensions means the sign-
            # gradient step below leaves those coordinates completely
            # unchanged, regardless of what the raw gradient says.
            grad = grad * self.mutable_mask

            with torch.no_grad():
                X_adv = X_adv + config.alpha * grad.sign()

                # Project back into the epsilon-ball around the original X
                # (L_inf ball), a la Madry et al. PGD.
                delta = torch.clamp(X_adv - X, min=-config.epsilon, max=config.epsilon)
                X_adv = X + delta

                # --- Core constraint #2 & #3: integer rounding + non-negativity ---
                X_adv = self._apply_domain_constraints(X_adv)

            X_adv = X_adv.detach()

        return X_adv


@torch.no_grad()
def evasion_success_rate(model: nn.Module, X_adv: torch.Tensor, y_true: torch.Tensor,
                          device: torch.device) -> dict:
    """
    Evasion Success Rate = fraction of TRULY MALICIOUS flows that the
    model now (incorrectly) classifies as Benign after perturbation.

    This is the attacker's headline metric: "how often does my crafted
    traffic get past the NIDS?" It is distinct from plain adversarial
    accuracy because we specifically condition on samples that started
    out correctly flagged as malicious and were truly malicious to begin
    with (evading detection on an already-benign flow is not a security
    win for the attacker).
    """
    model.eval()
    logits = model(X_adv.to(device))
    preds = torch.argmax(logits, dim=1).cpu()

    malicious_mask = (y_true == 1)
    n_malicious = malicious_mask.sum().item()
    if n_malicious == 0:
        return {"evasion_success_rate": float("nan"), "n_malicious": 0, "n_evaded": 0}

    evaded = ((preds == 0) & malicious_mask).sum().item()
    rate = evaded / n_malicious

    print(f"[Attack] Evasion Success Rate: {rate:.2%}  ({evaded}/{n_malicious} malicious "
          f"flows misclassified as Benign after perturbation)")
    return {"evasion_success_rate": rate, "n_malicious": n_malicious, "n_evaded": evaded}


def craft_adversarial_malicious_set(attacker: ConstrainedTabularAttack, X: np.ndarray,
                                     y: np.ndarray, config: AttackConfig,
                                     device: torch.device, batch_size: int = 256):
    """
    Convenience wrapper: takes raw numpy (X standardized, y labels), filters
    to malicious (y==1) samples only (evasion attacks only make sense on
    flows that are actually malicious), and returns adversarial examples
    for the whole malicious subset, batched to avoid memory blowups on
    large datasets.
    """
    mal_idx = np.where(y == 1)[0]
    X_mal = torch.tensor(X[mal_idx], dtype=torch.float32)
    y_mal = torch.tensor(y[mal_idx], dtype=torch.long)

    adv_batches = []
    for i in range(0, len(X_mal), batch_size):
        xb = X_mal[i:i + batch_size]
        yb = y_mal[i:i + batch_size]
        adv_batches.append(attacker.generate(xb, yb, config).cpu())

    X_adv = torch.cat(adv_batches, dim=0)
    return X_adv, y_mal, mal_idx
