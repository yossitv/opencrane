# Project Configuration

> **Purpose:** This file defines environment-specific commands for agents.

---

## Tech Stack & Environment

### Language & Framework
- **Language:** Python 3.11+
- **Framework:** Streamlit (UI), PyTorch (ML)
- **Runtime:** Python on GB10 (DGX Spark)

### Manifest Files
- **Dependencies:** requirements.txt
- **Environment:** .env

### File Extensions
- **Main:** .py
- **Test:** test_*.py

---

## Standard Commands

### Dependency Management
```bash
# Install dependencies
pip install -r requirements.txt

# Or with uv
uv pip install -r requirements.txt
```

### Testing
```bash
pytest
```

### Development Server
```bash
# Start Streamlit dashboard
streamlit run edge/app.py

# Port
8501
```

### Training
```bash
# Setup training environment
bash setup_train.sh

# Run training
python training/train.py
```

### NemoClaw Setup
```bash
# Setup NemoClaw environment
bash startup.sh
```

---

**Last Updated:** 2026-03-18
**Created by:** /define-requirements command
