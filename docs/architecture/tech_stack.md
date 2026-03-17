# Tech Stack

## Language & Runtime
- Primary Language: Python 3.11+
- Runtime: Python (GB10 / Jetson Orin Nano)

## Framework
- Main Framework: Streamlit（ダッシュボード）
- ML Framework: PyTorch
- Additional: NumPy, Pandas, Matplotlib/Plotly（ヒートマップ）

## Frontend
- Framework: Streamlit（Python内蔵、フロントエンド別途不要）

## Backend/API
- Framework: Streamlit内蔵サーバー（デモ用、API分離不要）
- 推論: inference.py（直接呼び出し）

## Database
- Type: なし（デモはCSV/インメモリ）

## Infrastructure
- 学習: GB10 (DGX Spark)
- エッジ推論: Jetson Orin Nano 8GB（将来）
- デモ公開: ngrok
- エージェント実行: NemoClaw + OpenShell

## External Services
- Auth: なし（デモ）
- Model Registry: Hugging Face（将来、学習済みモデル公開用）
- NemoClaw: NVIDIA Cloud Inference（nemotron-3-super-120b-a12b）

## Development Tools
- Package Manager: pip / uv
- Testing: pytest（最小限）
- Linting: なし（4時間デモ）
- Formatter: なし（4時間デモ）

## Justification
- **Streamlit:** 4時間でダッシュボードを作るのに最速。Pythonだけで完結
- **PyTorch:** GB10/Jetsonとの互換性が高い。NVIDIAエコシステムと親和性大
- **NemoClaw:** ハッカソンの「Best Use of OpenClaw」賞を狙うため必須
- **CSV/インメモリ:** DB構築の時間を省く。デモには十分
