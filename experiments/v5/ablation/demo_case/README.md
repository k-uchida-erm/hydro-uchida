Case: demo_case
Started: 2025-09-09 01:35:19

## Planned changes / Notes
- 

## Diff summary
Changed files: 1, +1 / -1

## Changed files

| # | Path |
|---:|---|
| 1 | `src/config.py` |


## Key hunks

<details><summary>src/config.py</summary>

```diff
@@ -54,7 +54,7 @@
 # Loss weights (must match current good run)
 FIXED_WEIGHTS = {
     'pde': 1.0,
-    'bc': 40.0,
+    'bc': 30.0,
     'ic': 30.0,
     'obs': 8.0,
     'theta_obs': 40.0
```

</details>

## Key metrics

| Metric | Value |
|---|---:|
| RMSE | 5.756947e-02 |
| MAE | 3.800706e-02 |
| R^2 | 0.999235 |

| Condition | RMSE |
|---|---:|
| IC t=0 | BC |
| BC top | top(z=0.2) |

<details><summary>Per-time psi metrics</summary>

```text
t,  MAE,   RMSE,  MAE(deep z>=7), MAE(shallow z<=1), grad_MAE, z_front_true, z_front_pred, dz
  0.0, 0.0134, 0.0181, 0.0182, 0.0148, 0.0316, 8.00, 8.80, +0.80
  1.0, 0.0511, 0.0927, 0.1344, 0.0052, 0.1382, 9.80, 10.00, +0.20
  2.0, 0.0889, 0.1124, 0.1322, 0.0113, 0.1292, 9.30, 9.50, +0.20
  4.0, 0.0413, 0.0622, 0.0878, 0.0179, 0.0645, 8.20, 8.40, +0.20
 10.0, 0.0677, 0.0917, 0.0273, 0.0157, 0.0519, 0.20, 0.20, +0.00
 32.0, 0.0210, 0.0259, 0.0187, 0.0045, 0.0134, 0.20, 0.80, +0.60
```

</details>
