import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_loss_history(model_dir: Path):
    csv_path = model_dir / 'loss_history.csv'
    if not csv_path.exists():
        print(f"loss_history.csv not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12,8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for rar_iter in df['rar_iter'].unique():
        mask = df['rar_iter'] == rar_iter
        subset = df[mask]
        
        subset = subset.copy()
        subset['global_epoch'] = (rar_iter - 1) * 400 + subset['epoch']
        
        plt.plot(subset['global_epoch'], subset['total_loss'], 
                color=colors[rar_iter-1], alpha=0.7, linewidth=1,
                label=f'total (RAR {rar_iter})')
        plt.plot(subset['global_epoch'], subset['pde_loss'], 
                color=colors[rar_iter-1], alpha=0.5, linewidth=0.8,
                linestyle='--', label=f'pde (RAR {rar_iter})' if rar_iter == 1 else "")
        plt.plot(subset['global_epoch'], subset['ic_loss'], 
                color=colors[rar_iter-1], alpha=0.5, linewidth=0.8,
                linestyle=':', label=f'ic (RAR {rar_iter})' if rar_iter == 1 else "")
    
    plt.xlabel('Global Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss History (RAR Iterations)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    out = model_dir / 'plots' / 'loss_history.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved: {out}")
