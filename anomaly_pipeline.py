"""
anomaly_pipeline.py
--------------------
AE1 -> Isolation Forest -> AE2 sandwich ensemble.

Production hardening (v2):
  [MEM-1]  StandardScaler fitted with partial_fit in chunks — never allocates
           a second full (N x DIM) matrix. Training data is scaled in-place
           using chunked transform so peak RAM stays at one matrix, not two.

  [MEM-2]  train_autoencoder() accepts a numpy memmap — the DataLoader
           accesses rows on demand from disk instead of copying the entire
           dataset into a torch.Tensor in host RAM.

  [MEM-3]  Isolation Forest is fitted on a random subsample (default 200K rows)
           when the dataset exceeds that threshold.  score_samples() on the
           full latent set is done in chunks.

  [GPU-1]  Device selection order: CUDA -> MPS (Apple Silicon) -> CPU.
           Batch training automatically leverages available GPU VRAM.
           Falls back to CPU if no GPU is found without any code change.

  [RESUME] The pipeline saves a scaler checkpoint after Step 1 so a second
           crash does not force a full re-run from feature extraction.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("AnomalyPipeline")

# ── Subsample cap for Isolation Forest (rows) ─────────────────────────────────
_IF_MAX_ROWS = 200_000

# ── Chunk size for in-place scaling and chunked inference ─────────────────────
_SCALE_CHUNK = 50_000


# ══════════════════════════════════════════════════════════════════════════════
# Autoencoder
# ══════════════════════════════════════════════════════════════════════════════
class Autoencoder(nn.Module):
    """Fully connected Autoencoder with configurable depth."""

    def __init__(
        self,
        input_dim:   int        = 394,
        latent_dim:  int        = 64,
        hidden_dims: List[int]  = None,
        dropout:     float      = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(prev, h), nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),  nn.Dropout(dropout),
            ]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(prev, h), nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),  nn.Dropout(dropout),
            ]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ══════════════════════════════════════════════════════════════════════════════
# [MEM-2]  Disk-backed Dataset — reads rows from memmap without loading all RAM
# ══════════════════════════════════════════════════════════════════════════════
class NumpyDataset(Dataset):
    """
    Wraps a numpy ndarray OR memmap so DataLoader can iterate over it
    without materialising the whole thing as a torch.Tensor first.
    Each worker reads only the rows for its current batch.
    """

    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        # Returns a copy of the row as float32 tensor — only one row at a time
        return torch.tensor(self.data[idx], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Training utility
# ══════════════════════════════════════════════════════════════════════════════
def train_autoencoder(
    model:           Autoencoder,
    data:            np.ndarray,          # can be a memmap
    epochs:          int   = 30,
    batch_size:      int   = 512,
    lr:              float = 1e-3,
    device:          str   = "cpu",
    patience:        int   = 5,
    checkpoint_path: Optional[str] = None,
) -> List[float]:
    """
    Train with early stopping.  Uses NumpyDataset so the full data array
    is never converted to a single torch.Tensor in RAM.
    """
    model     = model.to(device)
    dataset   = NumpyDataset(data)
    loader    = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=0,   # 0 = no subprocess forking — safe on Windows
        pin_memory=(device != "cpu"),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=2, factor=0.5
    )
    criterion = nn.MSELoss()

    best_loss  = float("inf")
    no_improve = 0
    losses     = []
    N          = len(data)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(batch)

        avg_loss = epoch_loss / N
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:>3}/{epochs} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
    return losses


# ── Chunked reconstruction errors ────────────────────────────────────────────
@torch.no_grad()
def compute_reconstruction_errors(
    model:      Autoencoder,
    data:       np.ndarray,
    batch_size: int = 1024,
    device:     str = "cpu",
) -> np.ndarray:
    model.eval()
    model   = model.to(device)
    errors  = []
    dataset = NumpyDataset(data)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0)
    for batch in loader:
        batch = batch.to(device)
        x_hat, _ = model(batch)
        mse = ((batch - x_hat) ** 2).mean(dim=1).cpu().numpy()
        errors.append(mse)
    return np.concatenate(errors)


# ── Chunked latent vector extraction ─────────────────────────────────────────
@torch.no_grad()
def extract_latent_vectors(
    model:      Autoencoder,
    data:       np.ndarray,
    batch_size: int = 1024,
    device:     str = "cpu",
) -> np.ndarray:
    model.eval()
    model   = model.to(device)
    latents = []
    dataset = NumpyDataset(data)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0)
    for batch in loader:
        batch = batch.to(device)
        z     = model.encode(batch)
        latents.append(z.cpu().numpy())
    return np.concatenate(latents)


# ══════════════════════════════════════════════════════════════════════════════
# Device selection  [GPU-1]
# ══════════════════════════════════════════════════════════════════════════════
def _best_device() -> str:
    """
    CUDA -> MPS (Apple Silicon) -> CPU.
    Logs which device was chosen and why.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {name}  ({vram:.1f} GB VRAM)  -> using cuda")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS detected -> using mps")
        return "mps"
    logger.info("No GPU detected -> using cpu")
    return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Incremental StandardScaler helpers  [MEM-1]
# ══════════════════════════════════════════════════════════════════════════════
def _fit_scaler_chunked(
    data:       np.ndarray,
    chunk_size: int = _SCALE_CHUNK,
) -> StandardScaler:
    """
    Fit a StandardScaler incrementally using partial_fit so only one chunk
    is in working memory at a time.  Total extra RAM = chunk_size * DIM * 4 bytes.
    """
    scaler = StandardScaler()
    N = len(data)
    for start in range(0, N, chunk_size):
        scaler.partial_fit(data[start: start + chunk_size])
    return scaler


def _transform_inplace_chunked(
    data:       np.ndarray,
    scaler:     StandardScaler,
    chunk_size: int = _SCALE_CHUNK,
) -> np.ndarray:
    """
    Apply scaler transform in-place chunk by chunk.
    The original `data` array is modified and returned — no second full matrix.
    If data is a memmap or read-only, a float32 copy is made first.
    """
    if not data.flags["WRITEABLE"] or data.dtype != np.float32:
        data = data.astype(np.float32, copy=True)
    N = len(data)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        data[start:end] = scaler.transform(data[start:end])
    return data


# ══════════════════════════════════════════════════════════════════════════════
# IF chunked prediction  [MEM-3]
# ══════════════════════════════════════════════════════════════════════════════
def _if_predict_chunked(
    iso_forest:  IsolationForest,
    latent:      np.ndarray,
    chunk_size:  int = _SCALE_CHUNK,
) -> np.ndarray:
    """
    Run IsolationForest.predict() in chunks so the full latent matrix
    doesn't need to sit in RAM alongside the model's internal trees.
    """
    labels = np.empty(len(latent), dtype=np.int8)
    for start in range(0, len(latent), chunk_size):
        end = min(start + chunk_size, len(latent))
        labels[start:end] = iso_forest.predict(latent[start:end])
    return labels


def _if_score_chunked(
    iso_forest:  IsolationForest,
    latent:      np.ndarray,
    chunk_size:  int = _SCALE_CHUNK,
) -> np.ndarray:
    scores = np.empty(len(latent), dtype=np.float32)
    for start in range(0, len(latent), chunk_size):
        end = min(start + chunk_size, len(latent))
        scores[start:end] = iso_forest.score_samples(latent[start:end])
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# AnomalyPipeline
# ══════════════════════════════════════════════════════════════════════════════
class AnomalyPipeline:
    """
    Full AE1 -> IF -> AE2 sandwich ensemble.

    Memory-safe for 1M+ rows on machines with 8-16 GB RAM.
    Automatically uses GPU when available.
    """

    def __init__(
        self,
        input_dim:                   int   = 394,
        ae1_latent_dim:              int   = 64,
        ae2_latent_dim:              int   = 32,
        if_contamination:            float = 0.05,
        anomaly_threshold_percentile: float = 95.0,
        device:                      str   = "auto",
        model_dir:                   str   = "models",
    ):
        self.input_dim                    = input_dim
        self.ae1_latent_dim               = ae1_latent_dim
        self.ae2_latent_dim               = ae2_latent_dim
        self.if_contamination             = if_contamination
        self.anomaly_threshold_percentile = anomaly_threshold_percentile
        self.model_dir                    = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.device = _best_device() if device == "auto" else device

        self.ae1:              Optional[Autoencoder]     = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.ae2:              Optional[Autoencoder]     = None
        self.scaler:           Optional[StandardScaler]  = None
        self.ae2_threshold:    float                     = 0.0
        self._is_trained:      bool                      = False

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(
        self,
        data:       np.ndarray,
        ae1_epochs: int = 30,
        ae2_epochs: int = 40,
        batch_size: int = 512,
    ) -> Dict:
        """
        Full training pipeline — memory-safe for large datasets.

        Step 1: Incremental StandardScaler (partial_fit in chunks)        [MEM-1]
        Step 2: AE1 training via NumpyDataset (no full Tensor copy)       [MEM-2]
        Step 3: Extract AE1 latent vectors in chunks
        Step 4: Isolation Forest on subsample + chunked predict           [MEM-3]
        Step 5: AE2 training on normal subset only
        Step 6: Compute anomaly threshold on normal reconstruction errors

        The scaler is saved after Step 1 so a crash later doesn't require
        restarting from feature extraction.
        """
        N = len(data)
        logger.info(f"Training pipeline on {N:,} samples  (dim={self.input_dim})")
        logger.info(f"Device: {self.device}")

        # ── Step 1: Incremental scaling  [MEM-1] ──────────────────────────────
        scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists():
            logger.info("Step 1/5: Loading cached scaler (resuming after crash)")
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            data_scaled = _transform_inplace_chunked(data, self.scaler)
        else:
            logger.info("Step 1/5: Fitting scaler incrementally (chunked partial_fit)")
            self.scaler = _fit_scaler_chunked(data)
            # Save before transforming — crash-safe resume point
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info("  Scaler saved. Transforming data in-place ...")
            data_scaled = _transform_inplace_chunked(data, self.scaler)
        logger.info(f"  Scaling complete. data_scaled shape: {data_scaled.shape}")

        # ── Step 2: Train AE1  [MEM-2] ────────────────────────────────────────
        ae1_ckpt = str(self.model_dir / "ae1_checkpoint.pt")
        self.ae1 = Autoencoder(
            input_dim=self.input_dim,
            latent_dim=self.ae1_latent_dim,
            hidden_dims=[256, 128],
        )
        # Resume AE1 from checkpoint if it exists
        if Path(ae1_ckpt).exists():
            logger.info("Step 2/5: Loading AE1 from checkpoint (resuming)")
            self.ae1.load_state_dict(
                torch.load(ae1_ckpt, map_location=self.device)
            )
            ae1_losses = []
        else:
            logger.info("Step 2/5: Training AE1 (global representation)")
            ae1_losses = train_autoencoder(
                self.ae1, data_scaled,
                epochs=ae1_epochs, batch_size=batch_size,
                device=self.device, checkpoint_path=ae1_ckpt,
            )

        # ── Step 3: Extract AE1 latent vectors ────────────────────────────────
        logger.info("Step 3/5: Extracting AE1 latent vectors (chunked)")
        latent_vectors = extract_latent_vectors(
            self.ae1, data_scaled, batch_size=batch_size * 2, device=self.device
        )
        logger.info(f"  Latent shape: {latent_vectors.shape}")

        # ── Step 4: Isolation Forest on subsample  [MEM-3] ────────────────────
        if N > _IF_MAX_ROWS:
            logger.info(
                f"Step 4/5: Fitting IF on {_IF_MAX_ROWS:,}-row subsample "
                f"(dataset too large for full fit)"
            )
            rng     = np.random.default_rng(42)
            idx     = rng.choice(N, size=_IF_MAX_ROWS, replace=False)
            sample  = latent_vectors[idx]
        else:
            logger.info("Step 4/5: Fitting Isolation Forest on full latent space")
            sample  = latent_vectors

        self.isolation_forest = IsolationForest(
            contamination=self.if_contamination,
            n_estimators=200,
            max_samples=min(256, len(sample)),
            random_state=42,
            n_jobs=-1,
        )
        self.isolation_forest.fit(sample)

        logger.info("  Predicting anomaly labels (chunked) ...")
        if_labels = _if_predict_chunked(self.isolation_forest, latent_vectors)
        normal_mask = (if_labels == 1)
        n_normal    = int(normal_mask.sum())
        n_anomaly   = int((~normal_mask).sum())
        logger.info(f"  IF results -> Normal: {n_normal:,}  Anomalies: {n_anomaly:,}")

        if n_normal < 100:
            logger.warning(
                "Very few normal samples detected. "
                "Consider reducing if_contamination."
            )

        # ── Step 5: Train AE2 on normal subset ────────────────────────────────
        ae2_ckpt    = str(self.model_dir / "ae2_checkpoint.pt")
        data_normal = data_scaled[normal_mask]
        self.ae2    = Autoencoder(
            input_dim=self.input_dim,
            latent_dim=self.ae2_latent_dim,
            hidden_dims=[256, 128],
        )
        if Path(ae2_ckpt).exists():
            logger.info("Step 5/5: Loading AE2 from checkpoint (resuming)")
            self.ae2.load_state_dict(
                torch.load(ae2_ckpt, map_location=self.device)
            )
            ae2_losses = []
        else:
            logger.info(
                f"Step 5/5: Training AE2 on {n_normal:,} normal samples"
            )
            ae2_losses = train_autoencoder(
                self.ae2, data_normal,
                epochs=ae2_epochs, batch_size=batch_size,
                device=self.device, checkpoint_path=ae2_ckpt,
            )

        # ── Step 6: Anomaly threshold ──────────────────────────────────────────
        logger.info("  Computing AE2 anomaly threshold ...")
        normal_errors = compute_reconstruction_errors(
            self.ae2, data_normal,
            batch_size=batch_size * 2, device=self.device,
        )
        self.ae2_threshold = float(
            np.percentile(normal_errors, self.anomaly_threshold_percentile)
        )
        logger.info(
            f"AE2 threshold (p{self.anomaly_threshold_percentile:.0f}): "
            f"{self.ae2_threshold:.6f}"
        )

        self._is_trained = True
        self.save()

        return {
            "n_total":             N,
            "n_normal":            n_normal,
            "n_point_anomalies":   n_anomaly,
            "ae1_final_loss":      ae1_losses[-1] if ae1_losses else None,
            "ae2_final_loss":      ae2_losses[-1] if ae2_losses else None,
            "ae2_threshold":       self.ae2_threshold,
        }

    # ── Score ─────────────────────────────────────────────────────────────────
    def score(
        self,
        data:           np.ndarray,
        return_latent:  bool = False,
    ) -> Dict:
        if not self._is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")

        data_scaled = self.scaler.transform(data)

        ae1_errors = compute_reconstruction_errors(
            self.ae1, data_scaled, device=self.device
        )
        latent = extract_latent_vectors(
            self.ae1, data_scaled, device=self.device
        )
        if_scores  = _if_score_chunked(self.isolation_forest, latent)
        ae2_errors = compute_reconstruction_errors(
            self.ae2, data_scaled, device=self.device
        )

        is_anomaly    = ae2_errors > self.ae2_threshold
        normalised    = ae2_errors / (self.ae2_threshold + 1e-9)
        anomaly_score = np.clip(normalised / 2.0, 0.0, 1.0)

        result = {
            "ae1_errors":    ae1_errors,
            "ae2_errors":    ae2_errors,
            "if_scores":     if_scores,
            "is_anomaly":    is_anomaly,
            "anomaly_score": anomaly_score,
        }
        if return_latent:
            result["latent"] = latent
        return result

    # ── Fine-tune AE2 ─────────────────────────────────────────────────────────
    def fine_tune_ae2(
        self,
        new_normal_data: np.ndarray,
        epochs:          int = 10,
        batch_size:      int = 256,
    ):
        if not self._is_trained:
            raise RuntimeError("Pipeline must be trained before fine-tuning.")
        data_scaled = self.scaler.transform(new_normal_data)
        train_autoencoder(
            self.ae2, data_scaled,
            epochs=epochs, batch_size=batch_size,
            device=self.device,
            checkpoint_path=str(self.model_dir / "ae2_finetuned.pt"),
        )
        new_errors = compute_reconstruction_errors(
            self.ae2, data_scaled, device=self.device
        )
        self.ae2_threshold = float(
            np.percentile(new_errors, self.anomaly_threshold_percentile)
        )
        logger.info(f"Updated AE2 threshold: {self.ae2_threshold:.6f}")
        self.save()

    # ── Save ──────────────────────────────────────────────────────────────────
    def save(self):
        torch.save(self.ae1.state_dict(), self.model_dir / "ae1.pt")
        torch.save(self.ae2.state_dict(), self.model_dir / "ae2.pt")
        with open(self.model_dir / "isolation_forest.pkl", "wb") as f:
            pickle.dump(self.isolation_forest, f)
        with open(self.model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(self.model_dir / "pipeline_meta.pkl", "wb") as f:
            pickle.dump({
                "input_dim":                   self.input_dim,
                "ae1_latent_dim":              self.ae1_latent_dim,
                "ae2_latent_dim":              self.ae2_latent_dim,
                "if_contamination":            self.if_contamination,
                "ae2_threshold":               self.ae2_threshold,
                "anomaly_threshold_percentile": self.anomaly_threshold_percentile,
            }, f)
        logger.info(f"Pipeline saved to {self.model_dir}")

    # ── Load ──────────────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls, model_dir: str = "models", device: str = "auto"
    ) -> "AnomalyPipeline":
        model_dir = Path(model_dir)
        with open(model_dir / "pipeline_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        pipeline = cls(
            input_dim=meta["input_dim"],
            ae1_latent_dim=meta["ae1_latent_dim"],
            ae2_latent_dim=meta["ae2_latent_dim"],
            if_contamination=meta["if_contamination"],
            anomaly_threshold_percentile=meta["anomaly_threshold_percentile"],
            device=device,
            model_dir=str(model_dir),
        )
        pipeline.ae2_threshold = meta["ae2_threshold"]

        pipeline.ae1 = Autoencoder(
            input_dim=meta["input_dim"], latent_dim=meta["ae1_latent_dim"]
        )
        pipeline.ae1.load_state_dict(
            torch.load(model_dir / "ae1.pt", map_location=pipeline.device)
        )

        pipeline.ae2 = Autoencoder(
            input_dim=meta["input_dim"], latent_dim=meta["ae2_latent_dim"]
        )
        pipeline.ae2.load_state_dict(
            torch.load(model_dir / "ae2.pt", map_location=pipeline.device)
        )

        with open(model_dir / "isolation_forest.pkl", "rb") as f:
            pipeline.isolation_forest = pickle.load(f)
        with open(model_dir / "scaler.pkl", "rb") as f:
            pipeline.scaler = pickle.load(f)

        pipeline._is_trained = True
        logger.info(f"Pipeline loaded from {model_dir}")
        return pipeline
