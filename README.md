# OpenCrane

クレーンゲームの景品配置を、AI による難易度評価と利益率予測で支援するハッカソン向けプロジェクトです。

## 構成

- `startup.sh`: GB10 / Jetson 側で NemoClaw をセットアップ
- `setup_train.sh`: 学習用の依存関係をインストール
- `training/train.py`: 学習スクリプト
- `edge/fetch_camera_image.py`: エッジ側カメラ画像の取得

## GB10 での NemoClaw セットアップ

```bash
sshpass -p "nvidia" ssh -o StrictHostKeyChecking=no nvidia@100.86.124.57
cd ~/opencrane
git pull --ff-only origin main
export PATH="$HOME/.local/bin:$PATH"
OPENCRANE_NEMOCLAW_MODE=spark ./startup.sh
```

`startup.sh` 実行後、必要に応じて以下を使います。

```bash
export PATH="$HOME/.local/bin:$PATH"
nemoclaw setup-spark
nemoclaw onboard
nemoclaw list
nemoclaw <sandbox-name> connect
```

## 重要: NVIDIA API Key が必要

初回の `nemoclaw setup-spark` / `nemoclaw onboard` では、NVIDIA API key の入力が必要です。  
キーは `nvapi-...` 形式で、以下から取得します。

```text
https://build.nvidia.com/settings/api-keys
```

入力した認証情報は通常 `~/.nemoclaw/credentials.json` に保存されるため、毎回の再入力は不要です。

## 学習環境

```bash
python3 -m venv .venv
source .venv/bin/activate
./setup_train.sh
python training/train.py
```
