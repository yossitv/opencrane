import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch is not installed", allow_module_level=True)


def test_train_script_saves_model_and_metrics(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
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
        cwd=repo_root,
        env=env,
        check=True,
    )

    model_path = model_dir / "difficulty_model.pt"
    metrics_path = model_dir / "training_metrics.json"

    assert model_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["sample_count"] > 0
    assert metrics["model_path"] == str(model_path)
