# =============================================================================
# メインスクリプト
# =============================================================================
# このスクリプトは、PINNモデルの学習を実行します：
# 1. データの読み込み
# 2. モデルの初期化
# 3. 学習の実行
# 4. 結果の保存
# =============================================================================

import os
import sys
from datetime import datetime
import pytz
import subprocess

from src import PINN, DEVICE, train, load_all_data

def main():
    # モデルの初期化
    model = PINN().to(DEVICE)
    
    # データの読み込み
    soil_map, df_bc, X_ic, h0, df_obs = load_all_data()
    
    try:
        # モデルの学習
        model = train(model, soil_map, df_bc, X_ic, h0, df_obs)
        print("学習が完了しました")
    except KeyboardInterrupt:
        print("学習を中断しました。現在のモデルを保存します...")
    finally:
        print("バリデーションを自動実行します...")
        subprocess.run(["python3", "validate.py"])

if __name__ == "__main__":
    main()
