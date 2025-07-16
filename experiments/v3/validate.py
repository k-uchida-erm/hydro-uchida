import os
import torch
from src import PINN, DEVICE, load_validation, obs_loss

def get_latest_model_path():
    result_dir = "result"
    model_dirs = [d for d in os.listdir(result_dir) if d.startswith("model_")]
    if not model_dirs:
        raise FileNotFoundError("No model directory found in result/")
    latest_dir = sorted(model_dirs)[-1]
    return os.path.join(result_dir, latest_dir, "model.pt")

MODEL_PATH = get_latest_model_path()

def main():
    # モデルの初期化と重みロード
    model = PINN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # バリデーションデータの読み込み
    df_val = load_validation()

    # バリデーション損失の計算
    with torch.no_grad():
        val_loss = obs_loss(model, df_val)
    print(f"バリデーション損失: {val_loss.item():.4f}")

if __name__ == "__main__":
    main() 