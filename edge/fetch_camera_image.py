from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "opencrane-edge-captures"


@dataclass(frozen=True)
class EdgeConfig:
    host: str
    port: int
    user: str
    password: str
    camera_device: str
    gamma: float
    remote_agent_path: str
    remote_image_path: str
    output_dir: Path
    ssh_timeout: int


def read_env(*names: str, default: str | None = None, allow_empty: bool = False) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        if allow_empty:
            return value
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return default


def require_env(*names: str, allow_empty: bool = False) -> str:
    value = read_env(*names, allow_empty=allow_empty)
    if value is None:
        raise ValueError(f"Missing required env var: {' / '.join(names)}")
    return value


def parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Expected integer value, got {value!r}") from exc


def parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Expected float value, got {value!r}") from exc


def load_config() -> EdgeConfig:
    load_dotenv(ROOT_DIR / ".env")

    output_dir_raw = read_env(
        "edge_output_dir",
        "EDGE_OUTPUT_DIR",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    output_dir = Path(output_dir_raw or str(DEFAULT_OUTPUT_DIR))
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir

    return EdgeConfig(
        host=require_env("edge_host", "EDGE_HOST"),
        port=parse_int(read_env("edge_port", "EDGE_PORT", default="22"), 22),
        user=require_env("edge_user", "EDGE_USER"),
        password=require_env("edge_pass", "EDGE_PASS", allow_empty=True),
        camera_device=read_env("edge_camera_device", "EDGE_CAMERA_DEVICE", default="/dev/video0")
        or "/dev/video0",
        gamma=parse_float(read_env("edge_gamma", "EDGE_GAMMA", default="1.8"), 1.8),
        remote_agent_path=read_env(
            "edge_remote_agent_path",
            "EDGE_REMOTE_AGENT_PATH",
            default="/tmp/store_agent/main.py",
        )
        or "/tmp/store_agent/main.py",
        remote_image_path=read_env(
            "edge_remote_image_path",
            "EDGE_REMOTE_IMAGE_PATH",
            default="/tmp/store_agent/latest.jpg",
        )
        or "/tmp/store_agent/latest.jpg",
        output_dir=output_dir,
        ssh_timeout=parse_int(read_env("edge_ssh_timeout", "EDGE_SSH_TIMEOUT", default="15"), 15),
    )


def ssh_base_cmd(config: EdgeConfig) -> list[str]:
    return [
        "sshpass",
        "-p",
        config.password,
        "ssh",
        "-o",
        f"ConnectTimeout={config.ssh_timeout}",
        "-o",
        "StrictHostKeyChecking=no",
        "-p",
        str(config.port),
        f"{config.user}@{config.host}",
    ]


def scp_base_cmd(config: EdgeConfig) -> list[str]:
    return [
        "sshpass",
        "-p",
        config.password,
        "scp",
        "-o",
        f"ConnectTimeout={config.ssh_timeout}",
        "-o",
        "StrictHostKeyChecking=no",
        "-P",
        str(config.port),
    ]


def run_command(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip() or "command failed")
    return completed


def run_remote(config: EdgeConfig, remote_cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command([*ssh_base_cmd(config), remote_cmd], check=check)


def test_connection(config: EdgeConfig) -> None:
    completed = run_remote(config, "echo connected")
    if completed.stdout.strip() != "connected":
        raise RuntimeError(f"unexpected SSH response: {completed.stdout.strip()!r}")


def camera_status(config: EdgeConfig) -> str:
    completed = run_remote(
        config,
        f"ls {shlex.quote(config.camera_device)} >/dev/null 2>&1 && echo cam_ok || echo cam_missing",
    )
    return completed.stdout.strip()


def remote_file_exists(config: EdgeConfig, remote_path: str) -> bool:
    completed = run_remote(
        config,
        f"test -f {shlex.quote(remote_path)} && echo present || echo missing",
    )
    return completed.stdout.strip() == "present"


def remote_agent_exists(config: EdgeConfig) -> bool:
    return remote_file_exists(config, config.remote_agent_path)


def start_remote_agent(config: EdgeConfig) -> None:
    if not remote_agent_exists(config):
        raise RuntimeError(
            f"Remote capture script was not found: {config.remote_agent_path}. "
            "Deploy it from the old store-agents project or update edge_remote_agent_path."
        )

    remote_cmd = (
        "mkdir -p /tmp/store_agent && "
        "pkill -f 'python3 /tmp/store_agent/main.py' >/dev/null 2>&1 || true && "
        f"nohup python3 {shlex.quote(config.remote_agent_path)} "
        f"--device {shlex.quote(config.camera_device)} "
        f"--gamma {config.gamma} "
        "> /tmp/store_agent/stdout.log 2>&1 &"
    )
    run_remote(config, remote_cmd)


def wait_for_remote_image(config: EdgeConfig, *, timeout_seconds: int = 20) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if remote_file_exists(config, config.remote_image_path):
            return True
        time.sleep(1)
    return False


def fetch_latest_image(config: EdgeConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "latest.jpg"
    remote_source = f"{config.user}@{config.host}:{config.remote_image_path}"
    run_command([*scp_base_cmd(config), remote_source, str(output_path)])
    return output_path


def is_jpeg(image_path: Path) -> bool:
    with image_path.open("rb") as handle:
        return handle.read(2) == b"\xff\xd8"


def main() -> int:
    config = load_config()
    test_connection(config)

    cam_state = camera_status(config)
    image_exists = remote_file_exists(config, config.remote_image_path)
    started_remote_agent = False

    if not image_exists:
        start_remote_agent(config)
        started_remote_agent = True
        if not wait_for_remote_image(config):
            raise RuntimeError(
                f"Remote image did not appear at {config.remote_image_path} after starting the agent."
            )

    image_path = fetch_latest_image(config)
    if not is_jpeg(image_path):
        raise RuntimeError(f"Downloaded file is not a JPEG image: {image_path}")

    print(f"connected_host={config.host}")
    print(f"camera_status={cam_state}")
    print(f"started_remote_agent={str(started_remote_agent).lower()}")
    print(f"saved_image={image_path}")
    print(f"image_size_bytes={image_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
