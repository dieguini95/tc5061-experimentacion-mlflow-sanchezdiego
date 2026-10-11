# Resumen de corridas: tc5061_act3_fatiga_dia_siguiente

Corridas terminadas: 13. Ordenadas por cv_mae (menor es mejor).

| corrida | modelo | seed | alpha | n_estimators | max_depth | learning_rate | cv_mae | cv_mae_std | cv_mae_media | mae_prueba | rmse_prueba | mape_prueba | r2_prueba | train_mae | mae_media_prueba | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgb_depth_2 | xgboost | 42 |  | 200 | 2 | 0.08 | 1.4148 | 0.0170 | 2.2378 | 1.3891 | 1.7384 | 0.5755 | 0.5945 | 1.3775 | 2.2846 | 51a126fba849405488b1038c9b308b99 |
| ridge_alpha_0.01 | ridge | 42 | 0.01 |  |  |  | 1.4244 | 0.0167 | 2.2378 | 1.3984 | 1.7517 | 0.5837 | 0.5883 | 1.4223 | 2.2846 | 0d0ed250fc4f42a4a974ce57ef727e90 |
| ridge_baseline_repro | ridge | 42 | 1.0 |  |  |  | 1.4245 | 0.0167 | 2.2378 | 1.3985 | 1.7517 | 0.5838 | 0.5883 | 1.4223 | 2.2846 | 3d8dd7f4a54c4f99a31956a3f141b0e9 |
| ridge_baseline | ridge | 42 | 1.0 |  |  |  | 1.4245 | 0.0167 | 2.2378 | 1.3985 | 1.7517 | 0.5838 | 0.5883 | 1.4223 | 2.2846 | f9ff90a2a2fa49ffa7a9b55c0a0e1ec3 |
| ridge_seed_7 | ridge | 7 | 1.0 |  |  |  | 1.4246 | 0.0149 | 2.2337 | 1.4124 | 1.7611 | 0.6273 | 0.5894 | 1.4217 | 2.3174 | 7cf77db2e7d341cfbb8f8e49cbdba580 |
| xgb_baseline | xgboost | 42 |  | 200 | 4 | 0.08 | 1.4273 | 0.0121 | 2.2378 | 1.3920 | 1.7495 | 0.5745 | 0.5893 | 1.2870 | 2.2846 | df31b031b3c948f0bfccbbe0ba974739 |
| ridge_alpha_100 | ridge | 42 | 100.0 |  |  |  | 1.4275 | 0.0175 | 2.2378 | 1.4021 | 1.7532 | 0.5875 | 0.5875 | 1.4248 | 2.2846 | f318418824ad4e2b9dd96beaedd451d1 |
| xgb_lr_0.02 | xgboost | 42 |  | 200 | 4 | 0.02 | 1.4293 | 0.0151 | 2.2378 | 1.4108 | 1.7517 | 0.5948 | 0.5882 | 1.3766 | 2.2846 | 3c641b7ef8c646119db4fb1b620bae25 |
| xgb_n_est_50 | xgboost | 42 |  | 50 | 4 | 0.08 | 1.4310 | 0.0143 | 2.2378 | 1.4163 | 1.7557 | 0.5968 | 0.5864 | 1.3776 | 2.2846 | 843517a473144df9a614c2413d5d21a5 |
| xgb_n_est_600 | xgboost | 42 |  | 600 | 4 | 0.08 | 1.4591 | 0.0164 | 2.2378 | 1.4150 | 1.7884 | 0.5806 | 0.5708 | 1.1534 | 2.2846 | 9e8749039fe64d22a862e722de5dd7ac |
| ridge_alpha_1000 | ridge | 42 | 1000.0 |  |  |  | 1.4795 | 0.0207 | 2.2378 | 1.4532 | 1.7846 | 0.6269 | 0.5726 | 1.4644 | 2.2846 | ea90ff47bd75451c94160ba3e1410cba |
| xgb_lr_0.3 | xgboost | 42 |  | 200 | 4 | 0.3 | 1.4881 | 0.0131 | 2.2378 | 1.4300 | 1.8114 | 0.5801 | 0.5597 | 1.1241 | 2.2846 | c28055791e5d4c02a80dbd60a339fc6b |
| xgb_depth_8 | xgboost | 42 |  | 200 | 8 | 0.08 | 1.5050 | 0.0171 | 2.2378 | 1.4545 | 1.8526 | 0.5909 | 0.5394 | 0.7424 | 2.2846 | ad749f4be90047788ff17d2ff4bd5206 |

## Influencia de cada factor (un factor a la vez)

| factor | corridas | cv_mae_minimo | cv_mae_maximo | rango_cv_mae | rango_mae_prueba |
|---|---|---|---|---|---|
| max_depth (xgboost) | xgb_depth_2, xgb_baseline, xgb_depth_8 | 1.4148 | 1.5050 | 0.0903 | 0.0654 |
| learning_rate (xgboost) | xgb_lr_0.02, xgb_baseline, xgb_lr_0.3 | 1.4273 | 1.4881 | 0.0609 | 0.0380 |
| alpha (ridge) | ridge_alpha_0.01, ridge_baseline, ridge_alpha_100, ridge_alpha_1000 | 1.4244 | 1.4795 | 0.0550 | 0.0548 |
| n_estimators (xgboost) | xgb_n_est_50, xgb_baseline, xgb_n_est_600 | 1.4273 | 1.4591 | 0.0318 | 0.0243 |
| familia de modelo (ridge_baseline contra xgb_baseline) | ridge_baseline, xgb_baseline | 1.4245 | 1.4273 | 0.0028 | 0.0065 |
| semilla de partición (ridge) | ridge_baseline, ridge_seed_7 | 1.4245 | 1.4246 | 0.0001 | 0.0139 |
