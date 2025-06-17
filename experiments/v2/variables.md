# 変数・定数の使われ方まとめ

## 1. 設定値（src/config.py）

| 変数名         | 用途・説明                                                                 | 参照ファイル         |
|----------------|--------------------------------------------------------------------------|-----------------------|
| DEVICE         | 計算デバイス（GPU/CPU）                                                   | model.py, train.py, loader.py, loss.py |
| DTYPE          | データ型（torch.float32）                                                 | model.py, train.py, loader.py, loss.py |
| GRID           | 計算領域のグリッド情報（Nx, Ny, Nz, dx, dy, dz, Nt）                      | train.py              |
| DT             | 時間ステップ幅（秒）                                                      | train.py              |
| EPOCHS         | 学習エポック数                                                            | train.py              |
| BATCH_SIZE     | バッチサイズ（内部点の生成数）                                             | train.py              |
| LEARNING_RATE  | 学習率                                                                   | train.py              |
| MAX_GRAD_NORM  | 勾配クリッピングの閾値                                                    | train.py              |
| LOSS_WEIGHTS   | 各損失項の重み（pde, bc, ic, obs）                                        | train.py              |
| DATA_DIR       | データファイルのディレクトリパス                                           | loader.py             |

## 2. データ関連

| 変数名      | 用途・説明                                 | 参照ファイル      |
|-------------|------------------------------------------|--------------------|
| soil_map    | 土壌パラメータのDataFrame                 | main.py, train.py, loss.py  |
| df_bc       | 境界条件のDataFrame                       | main.py, train.py, loss.py  |
| X_ic        | 初期条件点のTensor                        | main.py, train.py, loss.py  |
| h0          | 初期水頭のTensor                          | main.py, train.py, loss.py  |
| df_obs      | 観測データのDataFrame                     | main.py, train.py, loss.py  |

## 3. 学習・推論

| 変数名      | 用途・説明                                 | 参照ファイル      |
|-------------|------------------------------------------|--------------------|
| model       | PINNモデル本体                            | main.py, train.py, loss.py, analysis/check_model.py, analysis/visualize.py  |
| X_int       | 内部点（ランダムサンプリング点）           | train.py, loss.py  |
| loss_pde    | PDE残差損失                              | train.py, loss.py  |
| loss_bc     | 境界条件損失                             | train.py, loss.py  |
| loss_ic     | 初期条件損失                             | train.py, loss.py  |
| loss_obs    | 観測データ損失                           | train.py, loss.py  |
| loss        | 総損失                                   | train.py           |

## 4. 出力ファイル

- result/model_YYYYMMDD_HHMMSS/model.pt : 学習済みモデル
- result/model_YYYYMMDD_HHMMSS/loss_history.csv : 損失履歴
- result/model_YYYYMMDD_HHMMSS/analysis_results.txt : モデル分析結果
- result/model_YYYYMMDD_HHMMSS/plots/ : 可視化画像

---

# メソッド・関数の使われ方まとめ

## 1. データローダー関連（src/loader.py）

| 関数名           | 用途・説明                                               | 参照ファイル      |
|------------------|--------------------------------------------------------|---------------------|
| load_all_data()  | すべてのデータ（soil, bc, ic, obs）をまとめて読み込む   | main.py, analysis/check_model.py, analysis/visualize.py |
| load_soil(path)  | 土壌パラメータをCSVから読み込む                         | loader.py           |
| get_soil_params(x, y, z, soil_map) | 指定座標の土壌パラメータを取得         | loss.py             |
| load_bc(path)    | 境界条件データをCSVから読み込む                         | loader.py           |
| load_ic(path)    | 初期条件データをCSVから読み込む                         | loader.py           |
| load_obs(path)   | 観測データをCSVから読み込む                             | loader.py           |

## 2. モデル関連（src/model.py）

| 関数名           | 用途・説明                                               | 参照ファイル      |
|------------------|--------------------------------------------------------|---------------------|
| PINN             | 物理情報ニューラルネットワークのクラス                  | main.py, train.py, analysis/check_model.py, analysis/visualize.py |
| Se(h, a, n, tr, ts) | 有効飽和度の計算（van Genuchten式）                | model.py, loss.py   |
| theta(h, a, n, tr, ts) | 含水比の計算                                    | model.py, loss.py   |
| dtheta_dh(h, a, n, tr, ts) | 含水比のh微分                               | model.py, loss.py   |
| K_unsat(h, a, n, tr, ts, Ks) | 不飽和透水係数の計算                      | model.py, loss.py   |

## 3. 損失関数関連（src/loss.py）

| 関数名           | 用途・説明                                               | 参照ファイル      |
|------------------|--------------------------------------------------------|---------------------|
| residual(model, X, soil_map, Ss) | 物理方程式の残差計算                   | train.py, loss.py   |
| bc_loss(model, df_bc) | 境界条件の損失計算                              | train.py, loss.py   |
| ic_loss(model, X_ic, h0) | 初期条件の損失計算                          | train.py, loss.py   |
| obs_loss(model, df_obs) | 観測データの損失計算                          | train.py, loss.py   |

## 4. 学習・分析・可視化（src/train.py, analysis/）

| 関数名           | 用途・説明                                               | 参照ファイル      |
|------------------|--------------------------------------------------------|---------------------|
| train(model, soil_map, df_bc, X_ic, h0, df_obs, epochs) | PINNモデルの学習 | main.py             |
| generate_internal_points() | PDE残差計算用の内部点をランダム生成           | train.py, loss.py   |
| run_analysis(model_dir)    | 学習後にモデル分析・可視化を実行              | train.py            |
| list_models(result_dir)    | 利用可能なモデルディレクトリ一覧を表示         | analysis/check_model.py, analysis/visualize.py |
| check_model(model_dir)     | モデルの状態・評価指標を表示                  | analysis/check_model.py      |
| plot_loss_history(loss_file, output_dir) | 損失履歴のグラフを作成         | analysis/visualize.py        |
| plot_prediction_vs_observation(model, df_obs, output_dir) | 予測と観測の比較グラフ | analysis/visualize.py        |
| plot_head_distribution(model, t, output_dir) | 水頭分布の3D可視化           | analysis/visualize.py        | 