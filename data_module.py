"""
Module 1 (v2): Data Preprocessing & Baseline NIDS Classifier
==============================================================

Changes in v2
--------------
* split_validation(): carves a stratified VALIDATION set out of the training
  data, so epoch/checkpoint selection never touches the test set.
* train_model(): cosine learning-rate decay (smooths the epoch-to-epoch swings
  seen with a constant learning rate) and best-epoch selection by validation F1.
* evaluate_model(): optional verbose=False so it can be called every epoch
  without flooding the console.

Security/ML rationale (unchanged)
----------------------------------
Network intrusion datasets (e.g. CICIDS2017/2018) are messy: Inf/NaN values
(from division-by-zero rate features such as Flow Bytes/s), mixed integer and
continuous columns, and heavy class imbalance. Cleaning must happen before
scaling, and the scaler must be fit on training data only. We also track which
columns are immutable / integer-valued / non-negative, because the constrained
attack engine (Module 2) needs that metadata to stay inside the feasible
region of real network traffic.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# --------------------------------------------------------------------------
# 1. Feature metadata
# --------------------------------------------------------------------------
@dataclass
class FeatureSchema:
    """
    Describes the semantics of each column so downstream modules know what is
    and isn't allowed to move.

    immutable_features:    attacker cannot change these (e.g. Destination Port).
    integer_features:      counts that must be whole numbers after perturbation.
    non_negative_features: values that can never be negative in reality.
    """
    feature_names: List[str]
    label_column: str
    immutable_features: List[str] = field(default_factory=list)
    integer_features: List[str] = field(default_factory=list)
    non_negative_features: List[str] = field(default_factory=list)

    def mutable_mask(self) -> np.ndarray:
        """1 = mutable / attackable, 0 = frozen, aligned to feature_names order."""
        return np.array(
            [0.0 if f in self.immutable_features else 1.0 for f in self.feature_names],
            dtype=np.float32,
        )

    def integer_mask(self) -> np.ndarray:
        return np.array(
            [1.0 if f in self.integer_features else 0.0 for f in self.feature_names],
            dtype=np.float32,
        )

    def non_negative_mask(self) -> np.ndarray:
        return np.array(
            [1.0 if f in self.non_negative_features else 0.0 for f in self.feature_names],
            dtype=np.float32,
        )


# Default constraint lists for CICIDS-style column names. Entries that are not
# in FEATURE_NAMES are ignored by build_schema() in main.py.
CICIDS_DEFAULT_IMMUTABLE = [
    "Destination Port",
    "Protocol",
]
CICIDS_DEFAULT_INTEGER = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
]
CICIDS_DEFAULT_NONNEG = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
]


# --------------------------------------------------------------------------
# 2. Preprocessing pipeline
# --------------------------------------------------------------------------
class NIDSPreprocessor:
    """
    Cleans a raw CICIDS-style CSV and produces model-ready arrays.

    Order matters: strip column names -> replace Inf with NaN and drop -> binary
    labels -> split BEFORE fitting the scaler (no test-set leakage) -> fit the
    StandardScaler on the training split only.
    """

    def __init__(self, schema: FeatureSchema, test_size: float = 0.2, random_state: int = 42):
        self.schema = schema
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    @staticmethod
    def _clean_label(series: pd.Series) -> pd.Series:
        """Map arbitrary label strings to a binary 0/1 encoding."""
        def to_binary(v):
            if isinstance(v, str):
                return 0 if v.strip().upper() == "BENIGN" else 1
            return int(v)
        return series.map(to_binary)

    def load_and_clean(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        keep_cols = self.schema.feature_names + [self.schema.label_column]
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing expected columns: {missing}")
        df = df[keep_cols].copy()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        n_before = len(df)
        df.dropna(inplace=True)
        n_after = len(df)
        print(f"[Preprocessor] Dropped {n_before - n_after} rows with NaN/Inf "
              f"({n_before} -> {n_after} rows).")

        df[self.schema.label_column] = self._clean_label(df[self.schema.label_column])
        return df

    def fit_transform_split(self, df: pd.DataFrame):
        """Returns X_train, X_test, y_train, y_test as numpy arrays."""
        X = df[self.schema.feature_names].values.astype(np.float32)
        y = df[self.schema.label_column].values.astype(np.int64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Fit scaler on the train split ONLY (prevents test-set leakage).
        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_test = self.scaler.transform(X_test).astype(np.float32)
        return X_train, X_test, y_train, y_test

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X).astype(np.float32)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X_scaled).astype(np.float32)


def split_validation(X_train: np.ndarray, y_train: np.ndarray, val_frac: float = 0.1, seed: int = 42):
    """
    Carve a stratified validation set out of the TRAINING data.

    The validation set is used to pick the best epoch (checkpoint selection).
    Selecting on the test set would make the reported test numbers optimistic,
    so the test set is only touched once, at the very end.

    Returns X_tr, X_val, y_tr, y_val.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=seed, stratify=y_train
    )
    return X_tr, X_val, y_tr, y_val


# --------------------------------------------------------------------------
# 3. PyTorch Dataset / DataLoader
# --------------------------------------------------------------------------
class TabularFlowDataset(Dataset):
    """Simple wrapper turning numpy arrays into a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_dataloaders(X_train, y_train, X_test, y_test, batch_size: int = 256):
    train_ds = TabularFlowDataset(X_train, y_train)
    test_ds = TabularFlowDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def make_eval_loader(X, y, batch_size: int = 1024):
    """Non-shuffled loader, used for the validation set."""
    return DataLoader(TabularFlowDataset(X, y), batch_size=batch_size, shuffle=False)


# --------------------------------------------------------------------------
# 4. Baseline MLP classifier
# --------------------------------------------------------------------------
class NIDSClassifier(nn.Module):
    """
    Compact MLP for tabular flow classification. BatchNorm stabilizes training
    on heavy-tailed traffic features, Dropout limits overfitting, and raw logits
    are returned so the same network works with CrossEntropyLoss and with the
    gradient-based attacks in Module 2.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = (128, 64, 32), num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------
# 5. Training / evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device,
                   tag: str = "Eval", verbose: bool = True):
    """
    Accuracy, Precision, Recall, F1 (binary, positive class = Malicious = 1).
    Returns a dict so callers can aggregate results. Set verbose=False to
    suppress printing (used for per-epoch validation).
    """
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in data_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        print(f"[{tag}] Accuracy={acc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
        print(f"[{tag}] Confusion matrix (rows=true, cols=pred):\n{cm}")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}


def train_model(model: nn.Module, train_loader: DataLoader, device: torch.device,
                epochs: int = 20, lr: float = 1e-3, class_weights: Optional[torch.Tensor] = None,
                val_loader: Optional[DataLoader] = None, verbose: bool = True):
    """
    Trains the baseline classifier.

    * Cosine learning-rate decay: large steps early, small steps late, which
      removes most of the epoch-to-epoch swings of a constant learning rate.
    * If val_loader is given, the epoch with the best validation F1 is kept and
      restored at the end (checkpoint selection on validation data, not test).
    * class_weights counter the benign-heavy imbalance typical of NIDS data.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_f1 = None
        if val_loader is not None:
            val_f1 = evaluate_model(model, val_loader, device, verbose=False)["f1"]
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            extra = f" | val F1: {val_f1:.4f}" if val_f1 is not None else ""
            print(f"[Train] Epoch {epoch:3d}/{epochs} - avg loss: {total_loss / n_batches:.4f}{extra}")

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"[Train] Restored best epoch by validation F1 ({best_f1:.4f})")
    return model


def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights, useful given typical NIDS imbalance."""
    classes, counts = np.unique(y_train, return_counts=True)
    weights = counts.sum() / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)
