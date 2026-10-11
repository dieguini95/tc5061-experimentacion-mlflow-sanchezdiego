#!/usr/bin/env bash
# Ejecuta las 13 corridas de la comparación dentro del mismo experimento de MLflow.
# Uso: bash run_experiments.sh   (funciona desde cualquier carpeta)
# Requiere el entorno con requirements.txt instalado. Para otro intérprete: PYTHON=/ruta/python bash run_experiments.sh
set -e
set -u
set -o pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

correr() {
  "$PYTHON" train.py "$@"
}

# Línea base de Ridge (coincide con el notebook de la Actividad 2)
correr --model ridge --alpha 1.0 --run-name ridge_baseline

# Ridge: variación de alpha (un factor a la vez)
correr --model ridge --alpha 0.01 --run-name ridge_alpha_0.01
correr --model ridge --alpha 100 --run-name ridge_alpha_100
correr --model ridge --alpha 1000 --run-name ridge_alpha_1000

# XGBoost: línea base y variación de profundidad, número de árboles y tasa de aprendizaje
correr --model xgboost --run-name xgb_baseline
correr --model xgboost --max-depth 2 --run-name xgb_depth_2
correr --model xgboost --max-depth 8 --run-name xgb_depth_8
correr --model xgboost --n-estimators 50 --run-name xgb_n_est_50
correr --model xgboost --n-estimators 600 --run-name xgb_n_est_600
correr --model xgboost --learning-rate 0.02 --run-name xgb_lr_0.02
correr --model xgboost --learning-rate 0.3 --run-name xgb_lr_0.3

# Reproducibilidad: misma configuración que ridge_baseline; las métricas deben ser idénticas
correr --model ridge --alpha 1.0 --run-name ridge_baseline_repro

# Sensibilidad a la partición: misma configuración con otra semilla
correr --model ridge --alpha 1.0 --seed 7 --run-name ridge_seed_7
