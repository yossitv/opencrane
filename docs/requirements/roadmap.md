# Project Roadmap

## Phase 1: ハッカソンデモ（4時間）

- [ ] `startup.sh` — NemoClaw環境セットアップスクリプト
- [ ] `setup_train.sh` — 学習環境セットアップスクリプト
- [ ] `training/train.py` — モデル学習コード（.env読み込み、/models保存）
- [ ] `edge/inference.py` — 推論コード（モデル読み込み → 難易度スコア算出）
- [ ] `edge/app.py` — Streamlitダッシュボード（利益率計算 + ヒートマップ表示）

## Phase 2: プロダクト化（デモ後）

- [ ] カメラ映像からのリアルタイム景品検出統合
- [ ] 重心推定モデルの本格学習（大規模データセット）
- [ ] Jetson Orin Nano 8GBへのモデル蒸留・デプロイ
- [ ] Hugging Faceへの学習済みモデル公開
- [ ] 複数店舗対応・データ永続化
