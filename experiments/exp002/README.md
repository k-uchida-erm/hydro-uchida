# 地下水流動シミュレーション用PINN実装

物理情報ニューラルネットワーク（PINN）を用いた地下水流動シミュレーションの実装です。

## プロジェクト構成

```
experiments/exp002/
├── src/                    # ソースコード
│   ├── __init__.py        # パッケージ初期化
│   ├── config.py          # 設定ファイル
│   ├── model.py           # ニューラルネットワークと物理モデル
│   ├── loss.py            # 損失関数
│   ├── loader.py          # データ読み込みユーティリティ
│   └── train.py           # 学習ループ
├── data/                   # データディレクトリ
│   ├── soil.csv           # 土壌パラメータ
│   ├── bc.csv             # 境界条件
│   ├── ic.csv             # 初期条件
│   └── obs.csv            # 観測データ
├── main.py                # エントリーポイント
└── README.md              # 本ファイル
```

## 必要要件

- Python 3.8以上
- PyTorch
- NumPy
- Pandas

## 使用方法

1. `data`ディレクトリに以下の入力データファイルを準備：
   - `soil.csv`: 土壌パラメータ
   - `bc.csv`: 境界条件
   - `ic.csv`: 初期条件
   - `obs.csv`: 観測データ（任意）

2. シミュレーションの実行：
   ```bash
   python main.py
   ```

## 主な機能

- 物理情報ニューラルネットワーク（PINN）の実装
- van Genuchten-Mualemモデルによる不飽和流動の計算
- 偏微分方程式の残差に対する自動微分
- 飽和帯と不飽和帯の自動判定と計算
- Dirichlet条件とNeumann条件の境界条件に対応 