from __future__ import annotations

import argparse
from pathlib import Path

from edge.inference import PredictionRequest, predict
from edge.vision import estimate_image_features, render_estimate_overlay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an annotated OpenCrane evaluation image.")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--prize-cost", type=float, default=1800.0)
    parser.add_argument("--play-cost", type=float, default=100.0)
    parser.add_argument("--target-margin", type=float, default=0.35)
    parser.add_argument("--weight-grams", type=float, default=260.0)
    parser.add_argument("--center-x", type=float, default=0.55)
    parser.add_argument("--center-y", type=float, default=0.55)
    parser.add_argument("--grip-width", type=float, default=0.70)
    return parser


def default_output_path(image_path: Path) -> Path:
    evaluations_dir = image_path.parent / "evaluations"
    return evaluations_dir / f"{image_path.stem}-evaluated.jpg"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    image_path = Path(args.image_path).expanduser().resolve()
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else default_output_path(image_path)
    )

    result = predict(
        PredictionRequest(
            prize_cost=args.prize_cost,
            play_cost=args.play_cost,
            target_margin=args.target_margin,
            weight_grams=args.weight_grams,
            center_x=args.center_x,
            center_y=args.center_y,
            grip_width=args.grip_width,
            image_path=str(image_path),
        )
    )

    estimate = estimate_image_features(image_path)
    render_estimate_overlay(
        image_path,
        output_path,
        estimate=estimate,
        title_lines=[
            f"difficulty={result.difficulty_score:.4f}",
            f"hit_probability={result.hit_probability:.4f}",
            f"expected_profit={result.expected_profit:.2f}",
        ],
    )

    print(f"output_image={output_path}")
    print(f"used_image_features={str(result.used_image_features).lower()}")
    print(f"difficulty_score={result.difficulty_score}")
    print(f"hit_probability={result.hit_probability}")
    print(f"expected_profit={result.expected_profit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
