# Adversarially Robust Network Intrusion Detection System (NIDS)

A PyTorch project that attacks a tabular network-flow classifier with
**constraint-aware adversarial evasion attacks** and hardens it with
**adversarial training**, measuring both the robustness gained and the price
paid in false alarms.

Dataset: CICIDS2017 ("MachineLearningCSV" release), binary labels
(0 = benign, 1 = malicious).

## Why this is not image-style adversarial ML

Network flows are not images: not every point in an epsilon-ball is a valid
flow. The attack engine therefore respects domain constraints:

| Constraint | Applied to | How |
|---|---|---|
| Immutable features | Destination Port (and Protocol, if present) | gradient zeroed every step |
| Integer features | packet counts, TCP flag counts | rounded in raw units after each step |
| Non-negative features | durations, lengths, rates, counts | clamped at 0 in raw units after each step |

The model works in standardized space, but constraints are enforced in raw
(pre-scaling) units, because "no negative packets" only means something there.

## Project structure

| File | Purpose |
|---|---|
| `build_dataset.py` | Combines all CICIDS2017 day files and draws a 20% stratified sample |
| `data_module.py` | Preprocessing, feature schema, dataset/loaders, MLP, baseline training (cosine LR decay, validation-based checkpoint selection) |
| `attack_module.py` | Constrained PGD/FGSM attack engine and evasion-success-rate helpers |
| `defense_module.py` | Adversarial training (clean warmup, epsilon ramp, gradient clipping, both classes perturbed, validation-based checkpoint selection) and hardened-model evaluation |
| `main.py` | End-to-end pipeline: train baseline, attack it, adversarially train, attack the hardened model |
| `robustness_checks.py` | Sanity checks: corrected ESR, transfer attack, strong multi-restart attack, false-positive rates |
| `experiments.py` | Multi-seed epsilon sweep with mean/std, CSV and plot output |
| `threshold_analysis.py` | Operating-point analysis: recall and ESR at matched false-positive rates |
| `ablation.py` | Training epsilon x adversarial ratio x network width ablation |
| `per_attack_analysis.py` | Recall and corrected ESR per CICIDS2017 attack type, clean and under attack |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

Features (16, from the MachineLearningCSV release, which has no `Protocol`
column) and constraint lists are defined in `main.py` and `data_module.py`.
Check them against your CSV headers if you use a different release.

## How to run (in this order)

```powershell
# 1. Build the dataset sample (all days combined, 20% stratified)
python build_dataset.py

# 2. Quick end-to-end pipeline
python main.py --csv "data\combined_sample.csv" --seed 0

# 3. Sanity checks; trains and saves baseline.pt / hardened.pt (run once WITHOUT --load)
python robustness_checks.py --csv "data\combined_sample.csv" --seed 0

# 4. Operating points at matched false-positive rates (same seed as step 3)
python threshold_analysis.py --csv "data\combined_sample.csv" --seed 0

# 5. Main results: 3-seed epsilon sweep for the main model (width 128)
python experiments.py --csv "data\combined_sample.csv" --width 128 --seeds 0 1 2

# 6. Optional, slow: ablation grid
python ablation.py --csv "data\combined_sample.csv" --seed 0

# 7. Per-attack-type analysis (reuses baseline.pt / hardened.pt from step 3, same seed)
python per_attack_analysis.py --csv "data\combined_sample.csv" --seed 0
```

Outputs: `sweep_results_w128.csv`, `epsilon_sweep_w128.png`,
`threshold_results.csv`, `threshold_tradeoff.png`, `ablation_results.csv`,
`ablation_curves.csv`, `ablation_curves.png`, `robustness_tradeoff.png`,
`per_attack_results.csv`, `per_attack_recall.png`.

## Evaluation protocol

* Split: 20% test; 10% of the remaining training data is held out for
  validation. The validation set is used for checkpoint selection and for
  choosing decision thresholds. The test set is used only for final numbers.
* Attack: 40-step constrained PGD, L-infinity budget epsilon in standardized
  units. Strong check: 100 steps with 5 random restarts.
* **Corrected evasion success rate (ESR):** the fraction of malicious flows
  the model detects when clean that are evaded after the attack. This avoids
  counting flows the model already missed without any attack.
* **Best-of-sweep ESR:** a flow counts as evaded if it is evaded at this
  epsilon or any smaller one. Epsilon balls are nested, so this can never
  decrease and is a valid lower bound on the true worst case.
* Hardened model: epsilon 0.3, 50% of attacked samples swapped for adversarial
  versions, both classes perturbed (so "perturbed" cannot predict
  "malicious"), class-weighted loss, 2 clean warmup epochs, epsilon ramp,
  cosine learning-rate decay.

## Main results (width 128, 3 seeds, mean +/- std)

| | Clean recall | False positives | Corrected ESR at eps 0.05 / 0.1 / 0.3 / 0.5 | Recall after attack |
|---|---|---|---|---|
| Baseline | 93.6% +/- 1.0 | 1.8% +/- 1.4 | 88.1 / 93.6 / 99.7 / 99.9% | 11.5 / 6.1 / 0.3 / 0.1% |
| Hardened | 96.5% +/- 0.3 | 9.6% +/- 0.2 | 0.6 / 0.8 / 1.5 / 17.9% | 95.9 / 95.8 / 95.1 / 79.2% |

Other findings:

* The hardened model's ESR rises with epsilon in every seed, and recall after
  attack stays below clean recall, so it is not merely detecting perturbation
  artifacts. A strong 100-step, 5-restart attack gave 1.19% ESR against 1.10%
  for the 40-step attack, and a transfer attack from the baseline gave 0.27%.
* The price is false alarms: 9.6% versus 1.8%. At a matched 5% false-positive
  rate the hardened model detects roughly 85-92% of clean malicious flows
  (baseline: 97-99%) and about 78-80% under attack (baseline: below 1%). These
  matched-rate figures vary by several points between runs of the same
  configuration, so treat them as ranges.
* Training epsilon matters most: models trained at epsilon 0.1 keep only
  43-53% recall under an epsilon-0.3 attack, against 74-81% for models trained
  at 0.2 or 0.3. Adversarial ratio and width show no consistent effect.
* Width 256 lowers false positives (7.4%) but two of three seeds showed ESR
  falling as epsilon grew, which is impossible for a perfect attacker and means
  the attack under-optimised at larger budgets. Width 128 is the main model.
* Per attack type (seed 0), evasion is concentrated in slow DoS attacks: the
  overall corrected ESR of 1.1% hides 59% for Slowhttptest, 20% for slowloris
  and 10% for GoldenEye. Neither model reliably detects SSH-Patator at the
  default threshold (about 50% recall), and the rare classes (Infiltration, SQL
  injection, Heartbleed) have too few test flows to judge.

## Limitations

* One dataset sample (20% of CICIDS2017), an MLP, and binary labels.
* PGD is a heuristic attacker, so reported ESR is a lower bound on the true
  worst case.
* Robustness holds up to the training budget (epsilon 0.3) but degrades beyond
  it (about 18% ESR at epsilon 0.5, with wide seed-to-seed spread).
* Three seeds only; the false-positive cost of hardening is substantial.
* Epsilon is in standardized units, so it does not map evenly to realistic
  changes in raw traffic across features.
