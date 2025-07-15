# PINN Groundwater Model

## ディレクトリ構成

```
experiments/v3/
├── src/                    # ソースコード
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── loss.py
│   ├── loader.py
│   └── train.py
├── analysis/               # 分析・可視化スクリプト
│   ├── __init__.py
│   ├── check_model.py
│   └── visualize.py
├── data/                   # データファイル
│   ├── bc.csv
│   ├── ic.csv
│   ├── soil.csv
│   ├── soil_types.csv
│   └── verfi_1.csv
├── result/                 # 結果保存ディレクトリ
│   └── model_YYYYMMDD_HHMMSS/
│       ├── model.pt
│       ├── loss_history.csv
│       ├── analysis_results.txt
│       └── plots/
│           ├── loss_history.png
│           ├── pred_vs_obs.png
│           └── head_distribution/
│               └── head_distribution_t*.png
├── main.py                 # 学習用スクリプト
├── validate.py             # バリデーション用スクリプト
├── README.md
└── variables.md
```

## 使用方法

### 1. 学習（Training）

```sh
python main.py
```
- `data/obs.csv` があればそれを観測データ（正解データ）として学習します。
- `data/obs.csv` が無い場合は `data/verfi_1.csv` を自動的に観測データとして学習に使います。

### 2. バリデーション（Validation）

```sh
python validate.py
```
- `data/verfi_1.csv` を使って学習済みモデルのバリデーション損失を計算します。
- `validate.py` の `MODEL_PATH` を変更すれば、任意の学習済みモデルで評価できます。

### 3. データについて
- `verfi_1.csv` だけでも「学習」と「バリデーション」両方に使えます。
- データが増えた場合は、学習用（obs.csv）とバリデーション用（verfi_1.csv）を分けて運用することも可能です。

### 4. モデルの確認

利用可能なモデルの一覧を表示：
```bash
make check EXP=exp002 ARGS="--list"
```

特定のモデルを確認：
```bash
make check EXP=exp002 ARGS="--model model_20240321_123456"
```

### 5. 結果の可視化

利用可能なモデルの一覧を表示：
```bash
make visualize EXP=exp002 ARGS="--list"
```

特定のモデルの結果を可視化：
```bash
make visualize EXP=exp002 ARGS="--model model_20240321_123456"
```

## 出力ファイル

各モデルの結果は、`result/model_YYYYMMDD_HHMMSS/`ディレクトリに保存されます：

- `model.pt`: 学習済みモデル
- `loss_history.csv`: 損失履歴
- `analysis_results.txt`: モデル分析結果
- `plots/`: 可視化結果
  - `loss_history.png`: 損失履歴のグラフ
  - `pred_vs_obs.png`: 予測値と観測値の比較
  - `head_distribution/`: 水頭分布の可視化
    - `head_distribution_t*.png`: 各時間点での水頭分布 
