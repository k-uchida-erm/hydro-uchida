# PINN Groundwater Model

## ディレクトリ構造

```
experiments/exp002/
├── src/                    # ソースコード
│   ├── __init__.py
│   ├── config.py          # 設定ファイル
│   ├── model.py           # モデル定義
│   ├── loss.py            # 損失関数
│   └── train.py           # 学習ループ
├── analysis/              # 分析・可視化スクリプト
│   ├── __init__.py
│   ├── check_model.py     # モデル確認
│   └── visualize.py       # 結果可視化
├── data/                  # データファイル
│   └── observation.csv
├── result/               # 結果保存ディレクトリ
│   └── model_YYYYMMDD_HHMMSS/
│       ├── model.pt
│       ├── loss_history.csv
│       ├── loss_history.png
│       ├── pred_vs_obs.png
│       └── head_distribution_t*.png
├── main.py              # メインスクリプト
└── README.md           # ドキュメント
```

## 使用方法

### 1. 学習の実行

```bash
make run EXP=exp002
```

### 2. モデルの確認

利用可能なモデルの一覧を表示：
```bash
make check EXP=exp002 ARGS="--list"
```

特定のモデルを確認：
```bash
make check EXP=exp002 ARGS="--model model_20240321_123456"
```

### 3. 結果の可視化

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
- `loss_history.png`: 損失履歴のグラフ
- `pred_vs_obs.png`: 予測値と観測値の比較
- `head_distribution_t*.png`: 各時間点での水頭分布 