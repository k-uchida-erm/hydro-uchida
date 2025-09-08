import os
import subprocess
from src import PINN, DEVICE, train
from src.loader import load_all_data


def main():
    model = PINN().to(DEVICE)
    soil_map, df_bc, X_ic, psi0, df_train, df_obs, X_res = load_all_data()
    try:
        model = train(model, soil_map, df_bc, X_ic, psi0, df_train, df_obs, X_res)
        print("学習が完了しました")
    except KeyboardInterrupt:
        print("学習を中断しました。現在のモデルを保存します...")
    finally:
        print("バリデーションを自動実行します...")
        result_dir = "result"
        if os.path.exists(result_dir):
            model_dirs = [d for d in os.listdir(result_dir) if d.startswith("model_")]
            if model_dirs:
                latest_model = sorted(model_dirs)[-1]
                print(f"最新モデル: {latest_model}")
                
                print("モデル評価を実行中...")
                subprocess.run(["python3", "analysis/check_model.py", "--model", latest_model])
                
                print("可視化を実行中...")
                subprocess.run(["python3", "analysis/visualize.py", "--model", latest_model, "--paper"])
                
                print("全ての処理が完了しました！")
            else:
                print("モデルディレクトリが見つかりません")
        else:
            print("resultディレクトリが見つかりません")


if __name__ == "__main__":
    main()