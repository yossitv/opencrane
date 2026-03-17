#!/usr/bin/env bash

set -euo pipefail

NODE_MAJOR_REQUIRED="${NODE_MAJOR_REQUIRED:-22}"
NEMOCLAW_REPO_URL="${NEMOCLAW_REPO_URL:-https://github.com/NVIDIA/NemoClaw.git}"
OPENSHELL_INSTALL_URL="${OPENSHELL_INSTALL_URL:-https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh}"
OPENCRANE_NEMOCLAW_MODE="${OPENCRANE_NEMOCLAW_MODE:-auto}"
OPENCRANE_SKIP_ONBOARD="${OPENCRANE_SKIP_ONBOARD:-0}"
INSTALL_USER="${SUDO_USER:-${USER:-$(id -un)}}"

info() {
  printf '\033[1;34m[INFO]\033[0m %s\n' "$*"
}

warn() {
  printf '\033[1;33m[WARN]\033[0m %s\n' "$*"
}

error() {
  printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || error "Required command not found: $1"
}

ensure_user_context() {
  if [[ "${EUID}" -eq 0 ]]; then
    error "Run this script as a regular user with sudo access, not as root."
  fi
}

sudo_run() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    require_command sudo
    sudo "$@"
  fi
}

node_major() {
  node --version 2>/dev/null | sed 's/^v//' | cut -d. -f1
}

npm_major() {
  npm --version 2>/dev/null | cut -d. -f1
}

refresh_shell() {
  hash -r

  if [[ -f "${HOME}/.local/bin/env" ]]; then
    # OpenShell installer writes a small env script here when needed.
    # shellcheck disable=SC1091
    . "${HOME}/.local/bin/env"
  fi
}

ensure_ubuntu() {
  if [[ ! -r /etc/os-release ]]; then
    error "This script expects Ubuntu 22.04+ or Ubuntu-based Jetson Linux."
  fi

  # shellcheck disable=SC1091
  . /etc/os-release

  if [[ "${ID:-}" != "ubuntu" ]]; then
    warn "Detected ${PRETTY_NAME:-unknown OS}. NemoClaw officially targets Ubuntu 22.04+."
  fi

  local version_id
  version_id="${VERSION_ID:-0}"
  if [[ "${ID:-}" == "ubuntu" ]]; then
    if ! awk "BEGIN { exit !(${version_id} >= 22.04) }"; then
      error "Ubuntu 22.04 or newer is required. Detected ${PRETTY_NAME:-unknown OS}."
    fi
  fi
}

ensure_base_packages() {
  info "Installing base packages..."
  sudo_run apt-get update
  sudo_run apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    lsb-release \
    python3 \
    software-properties-common
}

ensure_docker() {
  if command -v docker >/dev/null 2>&1; then
    info "Docker already installed: $(docker --version)"
  else
    info "Docker not found. Installing docker.io from apt..."
    sudo_run apt-get update
    sudo_run apt-get install -y docker.io
  fi

  if command -v systemctl >/dev/null 2>&1 \
    && ([[ -f /lib/systemd/system/docker.service ]] || [[ -f /usr/lib/systemd/system/docker.service ]]); then
    sudo_run systemctl enable --now docker
  fi

  if getent group docker >/dev/null 2>&1 && ! id -nG "${INSTALL_USER}" | grep -qw docker; then
    info "Adding ${INSTALL_USER} to docker group..."
    sudo_run usermod -aG docker "${INSTALL_USER}"
    warn "Docker group membership was updated. A logout/login or 'newgrp docker' may be required after this script."
  fi
}

ensure_node() {
  local current_node_major current_npm_major

  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    current_node_major="$(node_major || true)"
    current_npm_major="$(npm_major || true)"

    if [[ "${current_node_major}" =~ ^[0-9]+$ ]] && [[ "${current_npm_major}" =~ ^[0-9]+$ ]] \
      && (( current_node_major >= NODE_MAJOR_REQUIRED )) && (( current_npm_major >= 10 )); then
      info "Node.js runtime already satisfies requirement: node $(node --version), npm $(npm --version)"
      return
    fi

    warn "Existing Node.js runtime is too old: node $(node --version), npm $(npm --version)"
  else
    info "Node.js not found."
  fi

  info "Installing Node.js ${NODE_MAJOR_REQUIRED}.x from NodeSource..."
  curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR_REQUIRED}.x" | sudo_run bash -
  sudo_run apt-get install -y nodejs
  refresh_shell

  require_command node
  require_command npm
  info "Installed node $(node --version), npm $(npm --version)"
}

ensure_openshell() {
  if command -v openshell >/dev/null 2>&1; then
    info "OpenShell already installed: $(openshell --version 2>/dev/null || echo present)"
    return
  fi

  info "Installing OpenShell..."
  curl -LsSf "${OPENSHELL_INSTALL_URL}" | sh
  refresh_shell
  require_command openshell
  info "OpenShell installed: $(openshell --version 2>/dev/null || echo present)"
}

is_spark_host() {
  if [[ "${OPENCRANE_NEMOCLAW_MODE}" == "spark" ]]; then
    return 0
  fi

  if [[ "${OPENCRANE_NEMOCLAW_MODE}" == "standard" ]]; then
    return 1
  fi

  if [[ -r /sys/class/dmi/id/product_name ]] && grep -Eqi 'dgx spark|gb10' /sys/class/dmi/id/product_name; then
    return 0
  fi

  return 1
}

npm_global_install() {
  local prefix
  prefix="$(npm config get prefix)"

  if [[ -w "${prefix}" ]]; then
    npm install -g "$@"
  else
    sudo_run npm install -g "$@"
  fi
}

install_nemoclaw() {
  if command -v nemoclaw >/dev/null 2>&1; then
    info "NemoClaw already installed: $(command -v nemoclaw)"
    return
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT

  info "Cloning NemoClaw from GitHub..."
  git clone --depth 1 "${NEMOCLAW_REPO_URL}" "${tmp_dir}/NemoClaw"

  info "Installing NemoClaw CLI..."
  npm_global_install "${tmp_dir}/NemoClaw"
  refresh_shell
  require_command nemoclaw
  info "NemoClaw installed: $(command -v nemoclaw)"
}

setup_spark_if_needed() {
  if ! is_spark_host; then
    return
  fi

  info "DGX Spark mode detected. Running 'nemoclaw setup-spark'..."
  nemoclaw setup-spark
}

run_onboard() {
  if [[ "${OPENCRANE_SKIP_ONBOARD}" == "1" ]]; then
    warn "Skipping 'nemoclaw onboard' because OPENCRANE_SKIP_ONBOARD=1"
    return
  fi

  info "Starting NemoClaw onboarding..."
  nemoclaw onboard
}

print_summary() {
  cat <<'EOF'

Next steps:
  1. If Docker group membership changed, run 'newgrp docker' or sign in again.
  2. Re-open your shell if 'openshell' or 'nemoclaw' is not on PATH yet.
  3. Verify with:
       nemoclaw --help
       openshell --help

EOF
}

main() {
  ensure_user_context
  ensure_ubuntu
  ensure_base_packages
  ensure_docker
  ensure_node
  ensure_openshell
  install_nemoclaw
  setup_spark_if_needed
  run_onboard
  print_summary
}

main "$@"
