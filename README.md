# TC5061 — Actividad 3: experimentación con MLflow

Convierte el notebook baseline de la Actividad 2 (fatiga del día siguiente) en un script parametrizado que registra cada corrida en MLflow Tracking.

## Contenido

| Archivo | Para qué sirve |
|---|---|
| `train.py` | Entrena una corrida y la registra en MLflow |
| `run_experiments.sh` | Ejecuta las 13 corridas de la comparación |
| `export_results.py` | Exporta la tabla resumen de MLflow a `results/` |
| `requirements.txt` | Dependencias con versiones fijadas |
| `notebooks/01_exploracion.ipynb` | Exploración y limpieza de datos (sin entrenamiento) |
| `results/` | Tablas exportadas de MLflow |
| `evidencias/` | Capturas de la interfaz de MLflow |

## Instalación

Probado con Python 3.12 en macOS.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En macOS, XGBoost necesita OpenMP: `brew install libomp`.

## Datos

El script descarga el dataset *Sleep Debt & Screen Time: Late Night Phone Habits* (Samar Talwar, CC BY-SA 4.0) con `kagglehub`. Si Kaggle pide credenciales, configúralas según la documentación de `kagglehub`. Alternativa: `--data-path ruta/al/bedtime_screentime_sleep_debt.csv`.

## Ejecución

Sin argumentos entrena el Ridge baseline:

```bash
python train.py
```

Cambiar hiperparámetros sin editar el código:

```bash
python train.py --model xgboost --max-depth 2 --run-name xgb_depth_2
python train.py --help
```

| Argumento | Default | Descripción |
|---|---|---|
| `--model` | `ridge` | `ridge` o `xgboost` |
| `--alpha` | 1.0 | [ridge] penalización L2 |
| `--n-estimators` | 200 | [xgboost] número de árboles |
| `--max-depth` | 4 | [xgboost] profundidad máxima de cada árbol |
| `--learning-rate` | 0.08 | [xgboost] tasa de aprendizaje |
| `--seed` | 42 | semilla de la partición train/test y de XGBoost |
| `--test-size` | 0.2 | fracción del dataset para prueba |
| `--cv-folds` | 5 | pliegues de validación cruzada |
| `--experiment-name` | `tc5061_act3_fatiga_dia_siguiente` | experimento de MLflow |
| `--run-name` | generado | nombre de la corrida |
| `--data-path` | descarga con kagglehub | CSV local del dataset |
| `--tracking-uri` | `sqlite:///<repo>/mlflow.db` | base de MLflow |

Para ejecutar las 13 corridas de la comparación:

```bash
bash run_experiments.sh
```

## Ver los resultados en MLflow

Desde la carpeta del repo:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir `http://127.0.0.1:5000` y elegir el experimento `tc5061_act3_fatiga_dia_siguiente`. Para regenerar las tablas de `results/`, ejecutar `python export_results.py`.

## Reproducibilidad

- `--seed` (42 por defecto) controla la partición train/test y XGBoost. La validación cruzada es determinista.
- Versiones fijadas en `requirements.txt`.
- Con los valores por defecto, el MAE de prueba de Ridge es 1.398, igual que en el notebook de la Actividad 2.
- Para empezar de cero, borra `mlflow.db` y `mlartifacts/`.

## Cargar un modelo guardado

```python
import mlflow

modelo = mlflow.pyfunc.load_model("runs:/<run_id>/modelo")
predicciones = modelo.predict(datos)  # datos: DataFrame con las columnas de FEATURES
```
