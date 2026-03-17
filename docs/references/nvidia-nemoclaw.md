# NVIDIA NemoClaw - GitHub Reference

Source: https://github.com/NVIDIA/NemoClaw

## Overview

NVIDIA NemoClaw is an open-source toolkit that streamlines deployment and operation of **OpenClaw autonomous assistants** within secure sandboxed environments. It integrates the NVIDIA OpenShell runtime (part of the NVIDIA Agent Toolkit), enabling safe agent execution with inference processed through NVIDIA's cloud infrastructure.

**Status:** Alpha — interfaces and behaviors may evolve.

**License:** Apache 2.0

**Languages:** TypeScript (36.8%), Shell (30.0%), JavaScript (27.3%), Python (4.8%)

## Repository Structure

```
NemoClaw/
├── .agents/skills/       # Agent skill definitions
├── bin/                  # Executable scripts
├── docs/                 # Documentation
├── nemoclaw-blueprint/   # Blueprint orchestration (Python)
├── nemoclaw/             # Main plugin source (TypeScript)
├── scripts/              # Utility scripts
├── test/                 # Test suites
├── Dockerfile
├── Makefile
├── package.json
├── pyproject.toml
├── install.sh / uninstall.sh
└── spark-install.md      # DGX Spark deployment guide
```

## Architecture

### Components

| Component | Description |
|-----------|-------------|
| **Plugin Layer** | TypeScript CLI — launch, connect, status, log streaming for sandbox lifecycle |
| **Blueprint System** | Versioned Python artifacts — sandbox instantiation, policy enforcement, inference config |
| **Sandbox Environment** | Isolated OpenShell containers running OpenClaw with kernel-enforced restrictions |
| **Inference Routing** | Cloud model requests intercepted by OpenShell, routed to NVIDIA inference backends |

### Security Layers

| Layer | Function | Mutability |
|-------|----------|------------|
| **Network** | Restricts unauthorized outbound traffic | Dynamically reloadable |
| **Filesystem** | Confines access to `/sandbox` and `/tmp` | Immutable post-creation |
| **Process** | Prevents privilege escalation and hazardous syscalls | Immutable post-creation |
| **Inference** | Redirects model API calls through approved backends | Dynamically reloadable |

## Prerequisites

- Ubuntu 22.04 LTS or newer
- Node.js 20+ with npm 10+ (22 recommended)
- Docker running and configured
- NVIDIA OpenShell pre-installed

## Quick Setup

```bash
curl -fsSL https://nvidia.com/nemoclaw.sh | bash
```

This runs guided onboarding: installs Node.js if absent, establishes sandbox, configures inference, applies security policies.

Output on success:
```
──────────────────────────────────────────────────
Sandbox my-assistant (Landlock + seccomp + netns)
Model nvidia/nemotron-3-super-120b-a12b
──────────────────────────────────────────────────
```

## CLI Commands

### Host-Level

```bash
nemoclaw onboard                    # Setup wizard
nemoclaw <name> connect            # Interactive shell access
nemoclaw <name> status             # Health monitoring
nemoclaw <name> logs --follow      # Log streaming
openshell term                     # Monitoring TUI
```

### Inside Sandbox

```bash
openclaw tui                                    # Chat interface
openclaw agent --agent main --local -m "text"  # Single message execution
```

## Inference Configuration

- **Provider:** NVIDIA Cloud
- **Model:** `nvidia/nemotron-3-super-120b-a12b`
- **Auth:** API key from build.nvidia.com (requested during `nemoclaw onboard`)

## Dependencies

- **Runtime:** NVIDIA OpenShell
- **Frameworks:** TypeScript, Node.js, Python
- **Containerization:** Docker + kernel isolation (Landlock, seccomp, network namespace)
- **Package Managers:** npm, uv
