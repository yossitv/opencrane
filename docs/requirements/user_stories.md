# User Stories

## As a 店舗スタッフ

### Story 1: 利益率予測ダッシュボード
**As a** 店舗スタッフ
**I want** 景品の難易度スコアと利益率予測をダッシュボードで確認したい
**So that** 景品配置が儲かるかどうかを補充前に定量的に判断できる

**Acceptance Criteria:**
- [ ] 景品情報（単価、play単価、目標利益率）を入力できる
- [ ] 難易度スコア（0.0〜1.0）が表示される
- [ ] 利益率の予測値が表示される
- [ ] ヒートマップで取りやすさの分布が可視化される

### Story 2: トレーニング実行
**As a** 開発者
**I want** `train.py` でモデルを学習し `/training/models/` に保存したい
**So that** 新しいデータで難易度評価モデルを更新できる

**Acceptance Criteria:**
- [ ] `setup_train.sh` で学習環境が構築できる
- [ ] `.env` からハイパーパラメータを読み込める
- [ ] 学習完了後にモデルが `/training/models/` に保存される

### Story 3: NemoClaw環境構築
**As a** 開発者
**I want** `startup.sh` 一発でNemoClaw環境を構築したい
**So that** ハッカソン会場ですぐにOpenClaw環境が使える

**Acceptance Criteria:**
- [ ] `startup.sh` を実行するとNemoClawがインストールされる
- [ ] サンドボックス環境が立ち上がる
