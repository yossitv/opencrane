from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURE_NAMES = [
    "prize_cost",
    "play_cost",
    "target_margin",
    "weight_grams",
    "center_x",
    "center_y",
    "grip_width",
]
TARGET_NAME = "difficulty_score"


@dataclass
class TrainConfig:
    data_path: Path
    model_dir: Path
    model_name: str
    metrics_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_dim: int
    requested_device: str
    seed: int


def resolve_path(root_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return root_dir / candidate


def load_config() -> TrainConfig:
    root_dir = Path(__file__).resolve().parents[1]
    load_dotenv(root_dir / ".env")

    return TrainConfig(
        data_path=resolve_path(
            root_dir,
            os.getenv("TRAIN_DATA_PATH", "training/data/sample_training_data.csv"),
        ),
        model_dir=resolve_path(
            root_dir,
            os.getenv("TRAIN_MODEL_DIR", "training/models"),
        ),
        model_name=os.getenv("TRAIN_MODEL_NAME", "difficulty_model.pt"),
        metrics_name=os.getenv("TRAIN_METRICS_NAME", "training_metrics.json"),
        epochs=int(os.getenv("TRAIN_EPOCHS", "200")),
        batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "8")),
        learning_rate=float(os.getenv("TRAIN_LEARNING_RATE", "0.01")),
        hidden_dim=int(os.getenv("TRAIN_HIDDEN_DIM", "16")),
        requested_device=os.getenv("TRAIN_DEVICE", "cpu"),
        seed=int(os.getenv("TRAIN_SEED", "42")),
    )


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA was requested, but it is not available. Falling back to CPU.")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data was not found: {csv_path}")

    rows: list[list[float]] = []
    targets: list[float] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = set(FEATURE_NAMES + [TARGET_NAME]) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Training data is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            rows.append([float(row[name]) for name in FEATURE_NAMES])
            targets.append(float(row[TARGET_NAME]))

    if not rows:
        raise ValueError(f"Training data is empty: {csv_path}")

    features = torch.tensor(rows, dtype=torch.float32)
    labels = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    return features, labels


def normalize_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = features.mean(dim=0)
    stds = features.std(dim=0, unbiased=False)
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    return (features - means) / stds, means, stds


def build_model(input_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid(),
    )


def train_model(config: TrainConfig) -> tuple[Path, Path]:
    set_seed(config.seed)
    device = choose_device(config.requested_device)

    features, labels = load_dataset(config.data_path)
    features, means, stds = normalize_features(features)

    dataset = TensorDataset(features, labels)
    batch_size = min(config.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = build_model(features.shape[1], config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    last_loss = 0.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_features.size(0)

        last_loss = epoch_loss / len(dataset)
        if epoch == 1 or epoch % 50 == 0 or epoch == config.epochs:
            print(f"epoch={epoch:03d} loss={last_loss:.6f}")

    model.eval()
    with torch.no_grad():
        predictions = model(features.to(device)).cpu()
        mae = torch.mean(torch.abs(predictions - labels)).item()

    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_dir / config.model_name
    metrics_path = config.model_dir / config.metrics_name

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": features.shape[1],
            "hidden_dim": config.hidden_dim,
            "feature_names": FEATURE_NAMES,
            "feature_means": means.tolist(),
            "feature_stds": stds.tolist(),
            "device": str(device),
            "config": {
                **asdict(config),
                "data_path": str(config.data_path),
                "model_dir": str(config.model_dir),
            },
        },
        model_path,
    )

    metrics = {
        "final_loss": last_loss,
        "mae": mae,
        "sample_count": len(dataset),
        "data_path": str(config.data_path),
        "model_path": str(model_path),
        "device": str(device),
        "feature_names": FEATURE_NAMES,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return model_path, metrics_path


def main() -> int:
    config = load_config()
    model_path, metrics_path = train_model(config)
    print(f"saved_model={model_path}")
    print(f"saved_metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
