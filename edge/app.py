from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from edge.inference import PredictionRequest, load_config, predict

DEFAULT_CAPTURE_PATHS = [
    ROOT_DIR / "edge" / "captures" / "latest.jpg",
    Path(tempfile.gettempdir()) / "opencrane-edge-captures" / "latest.jpg",
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg-top: #f6f0e3;
          --bg-bottom: #d7c2a3;
          --panel: rgba(255, 248, 236, 0.9);
          --panel-strong: rgba(255, 250, 242, 0.96);
          --ink: #1d1a17;
          --muted: #6f6458;
          --accent: #c56a2e;
          --accent-deep: #8f3f11;
          --good: #235c39;
          --warn: #8a4f17;
          --bad: #8e2f2f;
        }

        [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(circle at top left, rgba(255,255,255,0.7), transparent 35%),
            linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
          color: var(--ink);
        }

        [data-testid="stSidebar"] {
          background: rgba(245, 236, 220, 0.82);
          border-right: 1px solid rgba(140, 92, 54, 0.18);
        }

        .block-container {
          max-width: 1200px;
          padding-top: 2rem;
          padding-bottom: 3rem;
        }

        h1, h2, h3 {
          font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
          color: var(--ink);
          letter-spacing: -0.02em;
        }

        p, li, label, span, div {
          color: var(--ink);
        }

        .oc-shell {
          background: linear-gradient(180deg, rgba(255,250,242,0.96), rgba(255,246,236,0.84));
          border: 1px solid rgba(139, 96, 56, 0.16);
          border-radius: 24px;
          padding: 1.25rem 1.4rem;
          box-shadow: 0 18px 40px rgba(83, 57, 35, 0.08);
          margin-bottom: 1rem;
        }

        .oc-kicker {
          text-transform: uppercase;
          letter-spacing: 0.16em;
          font-size: 0.72rem;
          color: var(--accent-deep);
          margin-bottom: 0.4rem;
        }

        .oc-note {
          color: var(--muted);
          font-size: 0.95rem;
          line-height: 1.55;
        }

        .oc-badge {
          display: inline-block;
          padding: 0.2rem 0.55rem;
          border-radius: 999px;
          font-size: 0.8rem;
          margin-right: 0.5rem;
          background: rgba(197, 106, 46, 0.12);
          color: var(--accent-deep);
          border: 1px solid rgba(197, 106, 46, 0.2);
        }

        .oc-status-good {
          color: var(--good);
          font-weight: 600;
        }

        .oc-status-bad {
          color: var(--bad);
          font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_key_value_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def run_camera_fetch() -> tuple[Path | None, dict[str, str], str | None]:
    script_path = ROOT_DIR / "edge" / "fetch_camera_image.py"
    if not script_path.exists():
        return None, {}, f"Camera fetch script was not found: {script_path}"

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip() or "Camera fetch failed."
        return None, {}, message

    payload = parse_key_value_output(completed.stdout)
    image_path_raw = payload.get("saved_image")
    image_path = Path(image_path_raw) if image_path_raw else None
    return image_path, payload, None


def first_existing_capture() -> Path | None:
    for candidate in DEFAULT_CAPTURE_PATHS:
        if candidate.exists():
            return candidate
    return None


def format_currency(value: float) -> str:
    return f"JPY {value:,.0f}"


def render_header() -> None:
    st.markdown(
        """
        <div class="oc-shell">
          <div class="oc-kicker">Hack For Impact Demo Dashboard</div>
          <h1 style="margin: 0 0 0.6rem 0;">OpenCrane Placement Evaluator</h1>
          <p class="oc-note" style="margin-bottom: 0.8rem;">
            Estimate claw-machine placement difficulty, visualize likely hot spots, and compare expected
            revenue against your target margin before a prize setup goes live.
          </p>
          <span class="oc-badge">Local inference ready</span>
          <span class="oc-badge">Mock fallback enabled</span>
          <span class="oc-badge">Image centroid auto-detect</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Path | None:
    st.sidebar.markdown("## Capture Input")
    st.sidebar.caption("Use the latest Jetson frame if you have edge connectivity available.")

    if st.sidebar.button("Fetch latest camera frame", use_container_width=True):
        with st.spinner("Fetching latest frame from edge device..."):
            image_path, payload, error_message = run_camera_fetch()
        if error_message:
            st.sidebar.error(error_message)
        else:
            st.sidebar.success("Camera frame updated.")
            if image_path is not None:
                st.session_state["capture_path"] = str(image_path)
            st.session_state["capture_payload"] = payload

    capture_path_raw = st.session_state.get("capture_path")
    capture_path = Path(capture_path_raw) if capture_path_raw else first_existing_capture()

    if capture_path and capture_path.exists():
        st.sidebar.image(str(capture_path), caption="Latest frame", use_container_width=True)
        st.sidebar.caption(str(capture_path))
    else:
        st.sidebar.info("No local capture found yet.")

    config = load_config()
    st.sidebar.markdown("## Runtime")
    st.sidebar.code(
        "\n".join(
            [
                f"model_path={config.model_path}",
                f"device={config.requested_device}",
                f"mock_fallback={config.allow_mock_fallback}",
            ]
        )
    )

    payload = st.session_state.get("capture_payload")
    if payload:
        st.sidebar.markdown("## Last Fetch Status")
        st.sidebar.json(payload)

    return capture_path if capture_path and capture_path.exists() else None


def render_inputs(capture_path: Path | None) -> tuple[bool, PredictionRequest]:
    with st.form("placement-evaluator"):
        intro_left, intro_right = st.columns([1.2, 1.0], gap="large")

        with intro_left:
            st.markdown("### Prize Economics")
            prize_cost = st.number_input("Prize cost (JPY)", min_value=100.0, value=2200.0, step=100.0)
            play_cost = st.number_input("Play cost (JPY)", min_value=10.0, value=100.0, step=10.0)
            target_margin = st.slider("Target margin", min_value=0.05, max_value=0.90, value=0.35, step=0.01)

        with intro_right:
            st.markdown("### Physical Placement")
            weight_grams = st.number_input("Prize weight (g)", min_value=50.0, value=350.0, step=10.0)
            grip_width = st.slider("Grip width", min_value=0.30, max_value=1.00, value=0.70, step=0.01)
            center_x = st.slider("Center X", min_value=0.00, max_value=1.00, value=0.55, step=0.01)
            center_y = st.slider("Center Y", min_value=0.00, max_value=1.00, value=0.55, step=0.01)

        if capture_path:
            st.caption(
                "If a camera frame is present, OpenCrane will try to auto-estimate center X / Y and grip width "
                "from the prize image. The sliders below stay available as fallback values."
            )

        submitted = st.form_submit_button("Evaluate placement", use_container_width=True, type="primary")

    request = PredictionRequest(
        prize_cost=float(prize_cost),
        play_cost=float(play_cost),
        target_margin=float(target_margin),
        weight_grams=float(weight_grams),
        center_x=float(center_x),
        center_y=float(center_y),
        grip_width=float(grip_width),
        image_path=str(capture_path) if capture_path else None,
    )
    return submitted, request


def render_results(result_payload: dict[str, object], request: PredictionRequest, capture_path: Path | None) -> None:
    difficulty_score = float(result_payload["difficulty_score"])
    hit_probability = float(result_payload["hit_probability"])
    expected_revenue = float(result_payload["expected_revenue"])
    expected_profit = float(result_payload["expected_profit"])
    achieved_margin = float(result_payload["achieved_margin"])
    expected_plays = float(result_payload["expected_plays"])
    target_plays = float(result_payload["target_plays"])
    used_mock_model = bool(result_payload["used_mock_model"])
    used_image_features = bool(result_payload.get("used_image_features"))
    margin_gap = achieved_margin - request.target_margin

    if used_mock_model:
        st.warning(
            f"Model checkpoint could not be used. Showing mock inference instead. "
            f"Reason: {result_payload.get('fallback_reason')}"
        )
    else:
        st.success(f"Loaded model from {result_payload.get('model_path')}")

    metrics = st.columns(6, gap="small")
    metrics[0].metric("Difficulty score", f"{difficulty_score:.2f}")
    metrics[1].metric("Hit probability", f"{hit_probability:.0%}")
    metrics[2].metric("Expected plays", f"{expected_plays:.1f}")
    metrics[3].metric("Target plays", f"{target_plays:.1f}")
    metrics[4].metric("Expected profit", format_currency(expected_profit))
    metrics[5].metric("Achieved margin", f"{achieved_margin:.0%}", delta=f"{margin_gap:+.0%}")

    status = "Margin target reached" if margin_gap >= 0 else "Below target margin"
    status_class = "oc-status-good" if margin_gap >= 0 else "oc-status-bad"
    st.markdown(
        f'<div class="oc-shell"><div class="{status_class}">{status}</div>'
        f'<p class="oc-note" style="margin: 0.45rem 0 0 0;">'
        f'Current target margin is {request.target_margin:.0%}. '
        f'This placement projects {achieved_margin:.0%} based on the current difficulty estimate.'
        f"</p></div>",
        unsafe_allow_html=True,
    )

    visual_left, visual_right = st.columns([1.15, 0.85], gap="large")
    with visual_left:
        st.markdown("### Difficulty Heatmap")
        heatmap = result_payload["heatmap"]
        heatmap_df = pd.DataFrame(
            heatmap,
            index=[f"Y{i}" for i in range(len(heatmap))],
            columns=[f"X{i}" for i in range(len(heatmap[0]))],
        )
        st.table(heatmap_df.style.format("{:.2f}").background_gradient(cmap="YlOrBr", axis=None))

    with visual_right:
        st.markdown("### Input Snapshot")
        st.json(
            {
                "request": result_payload["feature_values"],
                "expected_revenue": expected_revenue,
                "device": result_payload["device"],
                "used_mock_model": used_mock_model,
                "used_image_features": used_image_features,
                "image_feature_estimate": result_payload.get("image_feature_estimate"),
                "image_feature_reason": result_payload.get("image_feature_reason"),
            }
        )
        if capture_path:
            st.markdown("### Camera Context")
            st.image(str(capture_path), caption="Latest capture", use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="OpenCrane Placement Evaluator",
        layout="wide",
    )
    inject_styles()
    render_header()
    capture_path = render_sidebar()

    submitted, request = render_inputs(capture_path)
    if submitted:
        try:
            result = predict(request)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
        else:
            st.session_state["last_result"] = result.to_dict()
            st.session_state["last_request"] = {
                "prize_cost": request.prize_cost,
                "play_cost": request.play_cost,
                "target_margin": request.target_margin,
                "weight_grams": request.weight_grams,
                "center_x": request.center_x,
                "center_y": request.center_y,
                "grip_width": request.grip_width,
                "image_path": request.image_path,
            }

    result_payload = st.session_state.get("last_result")
    request_payload = st.session_state.get("last_request")
    if result_payload and request_payload:
        render_results(result_payload, PredictionRequest(**request_payload), capture_path)
    else:
        st.markdown(
            """
            <div class="oc-shell">
              <div class="oc-kicker">Ready</div>
              <h3 style="margin-top: 0;">Submit a placement scenario to see the projected result.</h3>
              <p class="oc-note" style="margin-bottom: 0;">
                This demo scores from structured features, and now uses the latest camera frame to auto-estimate
                centroid-like placement features when one is available.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
