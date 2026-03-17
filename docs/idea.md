opencrane

problem
景品補充によって人によって変わるから、クオリティが変わって、利益率が変わるという問題
取りやすさのシミュレーションから、置く場所を決定して、儲かりやすくするって感じかな？

solution
景品の特定 -> 重心の特定 -> 難易度の評価 -> ヒートマップで当たりやすさの評価 -> 全体としての当たりやすさの評価するモデルの構築
全体当たりやすさ✖︎景品の単価 = play単価✖︎回数✖︎予定利益率で儲かるかどうかを評価する


how
clawに出すため、setup.shを作成 jetson へ NemoClawを使って環境を構築
GB10で画像検出して評価できるモデルを蒸留して、jetson orion nano 8gbに乗るようにする
taining データの作成 -> hugging faceへ出す
nanoclawへ作成したモデルを繋げる



構成

./
setup.sh

./training
setup_train.sh   # (必要なら) 学習特化の重いライブラリを入れるスクリプト
train.py         # .envを読み込んで学習し、/modelに保存するメインコード
/models

./edge #training/modelsで作成したaiか、envで設定するモデルをclawに入れて、動かす
app.py #Streamlitで作る店舗用ダッシュボード（UI）　実店舗のカメラから状態を参照して、景品の重心とかの評価から、いくらぐらい稼げそうか表示される状態
(demoでは ngrokとか使って、localhostを繋いで表示させる)
inference.py　#/training/model/ からモデルを読んで推論するコード


docs
https://github.com/NVIDIA/NemoClaw
https://github.com/NVIDIA/dgx-spark-playbooks
