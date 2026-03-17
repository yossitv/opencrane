import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch is not installed", allow_module_level=True)
if importlib.util.find_spec("PIL") is None:
    pytest.skip("Pillow is not installed", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw

from edge.inference import InferenceConfig, PredictionRequest, predict
from edge.vision import estimate_image_features, render_estimate_overlay


def train_test_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "models"

    env = os.environ.copy()
    env.update(
        {
            "TRAIN_MODEL_DIR": str(model_dir),
            "TRAIN_EPOCHS": "10",
            "TRAIN_DEVICE": "cpu",
        }
    )

    subprocess.run(
        [sys.executable, "training/train.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    return model_dir / "difficulty_model.pt"


def test_predict_uses_trained_model_checkpoint(tmp_path: Path) -> None:
    model_path = train_test_model(tmp_path)

    result = predict(
        PredictionRequest(
            prize_cost=1800.0,
            play_cost=100.0,
            target_margin=0.35,
            weight_grams=260.0,
            center_x=0.52,
            center_y=0.54,
            grip_width=0.78,
        ),
        config=InferenceConfig(
            model_path=model_path,
            requested_device="cpu",
            allow_mock_fallback=True,
        ),
    )

    assert result.used_mock_model is False
    assert result.model_path == str(model_path)
    assert result.device == "cpu"
    assert 0.0 <= result.difficulty_score <= 1.0
    assert len(result.heatmap) == 5
    assert all(len(row) == 5 for row in result.heatmap)


def test_predict_falls_back_to_mock_model_when_checkpoint_is_missing(tmp_path: Path) -> None:
    result = predict(
        PredictionRequest(
            prize_cost=2500.0,
            play_cost=100.0,
            target_margin=0.40,
        ),
        config=InferenceConfig(
            model_path=tmp_path / "missing-model.pt",
            requested_device="cpu",
            allow_mock_fallback=True,
        ),
    )

    assert result.used_mock_model is True
    assert result.model_path is None
    assert result.device == "mock"
    assert result.fallback_reason is not None
    assert 0.0 <= result.difficulty_score <= 1.0


def test_predict_uses_image_features_when_image_is_available(tmp_path: Path) -> None:
    model_path = train_test_model(tmp_path)
    image_path = tmp_path / "scene.jpg"

    image = Image.new("RGB", (240, 160), (18, 20, 26))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 104, 72, 154), fill=(64, 160, 72))
    draw.rectangle((150, 88, 226, 156), fill=(246, 78, 90))
    image.save(image_path)

    result = predict(
        PredictionRequest(
            prize_cost=1800.0,
            play_cost=100.0,
            target_margin=0.35,
            weight_grams=260.0,
            center_x=0.10,
            center_y=0.10,
            grip_width=0.30,
            image_path=str(image_path),
        ),
        config=InferenceConfig(
            model_path=model_path,
            requested_device="cpu",
            allow_mock_fallback=True,
        ),
    )

    assert result.used_image_features is True
    assert result.image_feature_reason is None
    assert result.image_feature_estimate is not None
    assert result.feature_values["center_x"] > 0.55
    assert result.feature_values["center_y"] > 0.60
    assert result.feature_values["grip_width"] >= 0.30


def test_render_estimate_overlay_writes_output_image(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.jpg"
    output_path = tmp_path / "evaluations" / "scene-evaluated.jpg"

    image = Image.new("RGB", (200, 140), (12, 16, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 80, 188, 136), fill=(235, 92, 110))
    image.save(image_path)

    estimate = estimate_image_features(image_path)
    saved_path = render_estimate_overlay(
        image_path,
        output_path,
        estimate=estimate,
        title_lines=["difficulty=0.7123"],
    )

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
