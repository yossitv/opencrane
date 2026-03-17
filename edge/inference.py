from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from edge.vision import estimate_image_features

DEFAULT_FEATURE_NAMES = [
    "prize_cost",
    "play_cost",
    "target_margin",
    "weight_grams",
    "center_x",
    "center_y",
    "grip_width",
]
DEFAULT_FEATURE_VALUES = {
    "prize_cost": 2200.0,
    "play_cost": 100.0,
    "target_margin": 0.35,
    "weight_grams": 350.0,
    "center_x": 0.55,
    "center_y": 0.55,
    "grip_width": 0.70,
}


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path
    requested_device: str
    allow_mock_fallback: bool


@dataclass(frozen=True)
class PredictionRequest:
    prize_cost: float
    play_cost: float
    target_margin: float
    weight_grams: float = DEFAULT_FEATURE_VALUES["weight_grams"]
    center_x: float = DEFAULT_FEATURE_VALUES["center_x"]
    center_y: float = DEFAULT_FEATURE_VALUES["center_y"]
    grip_width: float = DEFAULT_FEATURE_VALUES["grip_width"]
    image_path: str | None = None


@dataclass(frozen=True)
class PredictionResult:
    difficulty_score: float
    hit_probability: float
    expected_revenue: float
    expected_profit: float
    achieved_margin: float
    expected_plays: float
    target_plays: float
    heatmap: list[list[float]]
    feature_values: dict[str, float]
    used_mock_model: bool
    fallback_reason: str | None
    model_path: str | None
    device: str
    used_image_features: bool
    image_feature_reason: str | None
    image_feature_estimate: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    device: torch.device
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]
    model_path: Path


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def resolve_path(root_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return root_dir / candidate


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA was requested, but it is not available. Falling back to CPU.")
    return torch.device("cpu")


def build_model(input_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid(),
    )


def load_config() -> InferenceConfig:
    load_dotenv(ROOT_DIR / ".env")

    model_path_raw = os.getenv("INFERENCE_MODEL_PATH")
    if model_path_raw:
        model_path = resolve_path(ROOT_DIR, model_path_raw)
    else:
        model_dir = resolve_path(ROOT_DIR, os.getenv("TRAIN_MODEL_DIR", "training/models"))
        model_name = os.getenv("INFERENCE_MODEL_NAME", os.getenv("TRAIN_MODEL_NAME", "difficulty_model.pt"))
        model_path = model_dir / model_name

    return InferenceConfig(
        model_path=model_path,
        requested_device=os.getenv("INFERENCE_DEVICE", os.getenv("TRAIN_DEVICE", "cpu")),
        allow_mock_fallback=parse_bool(os.getenv("INFERENCE_ALLOW_MOCK"), True),
    )


@lru_cache(maxsize=8)
def load_model_bundle(model_path_raw: str, requested_device: str) -> ModelBundle:
    model_path = Path(model_path_raw)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint was not found: {model_path}")

    device = choose_device(requested_device)
    checkpoint = torch.load(model_path, map_location=device)

    feature_names = tuple(checkpoint.get("feature_names") or DEFAULT_FEATURE_NAMES)
    input_dim = int(checkpoint.get("input_dim", len(feature_names)))
    hidden_dim = int(checkpoint.get("hidden_dim", 16))
    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        raise ValueError(f"Checkpoint is missing model_state_dict: {model_path}")

    feature_means = tuple(
        float(value)
        for value in checkpoint.get(
            "feature_means",
            [DEFAULT_FEATURE_VALUES.get(name, 0.0) for name in feature_names],
        )
    )
    feature_stds = tuple(
        float(value)
        for value in checkpoint.get(
            "feature_stds",
            [1.0 for _ in feature_names],
        )
    )

    model = build_model(input_dim, hidden_dim)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    return ModelBundle(
        model=model,
        device=device,
        feature_names=feature_names,
        feature_means=feature_means,
        feature_stds=feature_stds,
        model_path=model_path,
    )


def request_to_features(request: PredictionRequest) -> dict[str, float]:
    return {
        "prize_cost": float(request.prize_cost),
        "play_cost": float(request.play_cost),
        "target_margin": float(request.target_margin),
        "weight_grams": float(request.weight_grams),
        "center_x": float(request.center_x),
        "center_y": float(request.center_y),
        "grip_width": float(request.grip_width),
    }


def maybe_apply_image_features(
    request: PredictionRequest,
) -> tuple[PredictionRequest, bool, str | None, dict[str, object] | None]:
    if not request.image_path:
        return request, False, None, None

    try:
        estimate = estimate_image_features(request.image_path)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        return request, False, str(exc), None

    return (
        PredictionRequest(
            prize_cost=request.prize_cost,
            play_cost=request.play_cost,
            target_margin=request.target_margin,
            weight_grams=request.weight_grams,
            center_x=estimate.center_x,
            center_y=estimate.center_y,
            grip_width=estimate.grip_width,
            image_path=request.image_path,
        ),
        True,
        None,
        estimate.to_dict(),
    )


def build_feature_tensor(feature_values: dict[str, float], bundle: ModelBundle) -> torch.Tensor:
    ordered_values = [feature_values.get(name, DEFAULT_FEATURE_VALUES.get(name, 0.0)) for name in bundle.feature_names]
    features = torch.tensor(ordered_values, dtype=torch.float32)
    means = torch.tensor(bundle.feature_means, dtype=torch.float32)
    stds = torch.tensor(bundle.feature_stds, dtype=torch.float32)
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    normalized = (features - means) / stds
    return normalized.unsqueeze(0).to(bundle.device)


def normalized_range(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0)


def mock_difficulty_score(request: PredictionRequest) -> float:
    price_penalty = normalized_range(request.prize_cost, 1000.0, 4000.0)
    play_bonus = normalized_range(request.play_cost, 100.0, 300.0)
    margin_penalty = normalized_range(request.target_margin, 0.20, 0.60)
    weight_penalty = normalized_range(request.weight_grams, 150.0, 600.0)
    center_offset = clamp(
        (abs(request.center_x - 0.5) + abs(request.center_y - 0.5)) / 0.5,
        0.0,
        1.0,
    )
    grip_bonus = normalized_range(request.grip_width, 0.50, 0.90)

    score = (
        0.30 * (1.0 - price_penalty)
        + 0.10 * play_bonus
        + 0.15 * (1.0 - margin_penalty)
        + 0.20 * (1.0 - weight_penalty)
        + 0.15 * (1.0 - center_offset)
        + 0.10 * grip_bonus
    )
    return clamp(score, 0.05, 0.95)


def build_heatmap(base_score: float, request: PredictionRequest, size: int = 5) -> list[list[float]]:
    if size < 2:
        raise ValueError("Heatmap size must be at least 2")

    spread = 0.12 + clamp(request.grip_width, 0.0, 1.0) * 0.18
    coords = [index / (size - 1) for index in range(size)]
    rows: list[list[float]] = []
    for y in coords:
        row: list[float] = []
        for x in coords:
            distance = math.sqrt((x - request.center_x) ** 2 + (y - request.center_y) ** 2)
            attenuation = math.exp(-((distance ** 2) / (2.0 * (spread ** 2))))
            cell_score = clamp(base_score * (0.35 + 0.65 * attenuation), 0.05, 0.95)
            row.append(round(cell_score, 4))
        rows.append(row)
    return rows


def derive_profit_metrics(
    request: PredictionRequest,
    difficulty_score: float,
) -> tuple[float, float, float, float, float]:
    # Difficulty is mapped into a low single-digit win-rate band so the
    # economics resemble a crane-game loop rather than a one-shot classifier.
    hit_probability = clamp(0.015 + difficulty_score * 0.045, 0.015, 0.08)
    expected_plays = 1.0 / hit_probability
    expected_revenue = request.play_cost * expected_plays
    expected_profit = expected_revenue - request.prize_cost
    achieved_margin = expected_profit / expected_revenue

    target_margin = clamp(request.target_margin, 0.0, 0.95)
    target_revenue = request.prize_cost / (1.0 - target_margin)
    target_plays = target_revenue / request.play_cost
    return hit_probability, expected_revenue, expected_profit, expected_plays, target_plays


def predict(request: PredictionRequest, config: InferenceConfig | None = None) -> PredictionResult:
    config = config or load_config()
    effective_request, used_image_features, image_feature_reason, image_feature_estimate = (
        maybe_apply_image_features(request)
    )
    feature_values = request_to_features(effective_request)

    used_mock_model = False
    fallback_reason: str | None = None
    model_path: str | None = None

    try:
        bundle = load_model_bundle(str(config.model_path), config.requested_device)
        feature_tensor = build_feature_tensor(feature_values, bundle)
        with torch.no_grad():
            difficulty_score = float(bundle.model(feature_tensor).item())
        difficulty_score = clamp(difficulty_score, 0.0, 1.0)
        device_name = str(bundle.device)
        model_path = str(bundle.model_path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        if not config.allow_mock_fallback:
            raise
        difficulty_score = mock_difficulty_score(effective_request)
        used_mock_model = True
        fallback_reason = str(exc)
        device_name = "mock"

    hit_probability, expected_revenue, expected_profit, expected_plays, target_plays = derive_profit_metrics(
        effective_request,
        difficulty_score,
    )

    return PredictionResult(
        difficulty_score=round(difficulty_score, 4),
        hit_probability=round(hit_probability, 4),
        expected_revenue=round(expected_revenue, 2),
        expected_profit=round(expected_profit, 2),
        achieved_margin=round(expected_profit / expected_revenue, 4),
        expected_plays=round(expected_plays, 2),
        target_plays=round(target_plays, 2),
        heatmap=build_heatmap(difficulty_score, effective_request),
        feature_values=feature_values,
        used_mock_model=used_mock_model,
        fallback_reason=fallback_reason,
        model_path=model_path,
        device=device_name,
        used_image_features=used_image_features,
        image_feature_reason=image_feature_reason,
        image_feature_estimate=image_feature_estimate,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenCrane difficulty inference.")
    parser.add_argument("--prize-cost", type=float, required=True)
    parser.add_argument("--play-cost", type=float, required=True)
    parser.add_argument("--target-margin", type=float, required=True)
    parser.add_argument("--weight-grams", type=float, default=DEFAULT_FEATURE_VALUES["weight_grams"])
    parser.add_argument("--center-x", type=float, default=DEFAULT_FEATURE_VALUES["center_x"])
    parser.add_argument("--center-y", type=float, default=DEFAULT_FEATURE_VALUES["center_y"])
    parser.add_argument("--grip-width", type=float, default=DEFAULT_FEATURE_VALUES["grip_width"])
    parser.add_argument("--image-path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = predict(
        PredictionRequest(
            prize_cost=args.prize_cost,
            play_cost=args.play_cost,
            target_margin=args.target_margin,
            weight_grams=args.weight_grams,
            center_x=args.center_x,
            center_y=args.center_y,
            grip_width=args.grip_width,
            image_path=args.image_path,
        )
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
