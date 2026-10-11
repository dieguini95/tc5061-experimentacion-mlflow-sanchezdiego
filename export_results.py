"""Exporta el resumen de las corridas del experimento para el análisis.

Escribe en results/:
- runs_summary.csv: una fila por corrida, ordenada por cv_mae (menor es mejor).
- runs_summary.md: la misma tabla en Markdown, más la tabla de influencia.
- influencia_hiperparametros.csv: cuánto cambia el MAE al mover cada factor, un factor a la vez.

Uso:
    python export_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from train import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, REPO_DIR

# Columnas de MLflow que se exportan, con su nombre corto.
COLUMNAS = {
    "tags.mlflow.runName": "corrida",
    "params.modelo": "modelo",
    "params.seed": "seed",
    "params.alpha": "alpha",
    "params.n_estimators": "n_estimators",
    "params.max_depth": "max_depth",
    "params.learning_rate": "learning_rate",
    "metrics.cv_mae": "cv_mae",
    "metrics.cv_mae_std": "cv_mae_std",
    "metrics.cv_mae_media": "cv_mae_media",
    "metrics.mae": "mae_prueba",
    "metrics.rmse": "rmse_prueba",
    "metrics.mape": "mape_prueba",
    "metrics.r2": "r2_prueba",
    "metrics.train_mae": "train_mae",
    "metrics.mae_media": "mae_media_prueba",
    "run_id": "run_id",
}

# Grupos de corridas que cambian un solo factor respecto de su línea base.
GRUPOS_INFLUENCIA = {
    "familia de modelo (ridge_baseline contra xgb_baseline)": ["ridge_baseline", "xgb_baseline"],
    "alpha (ridge)": ["ridge_alpha_0.01", "ridge_baseline", "ridge_alpha_100", "ridge_alpha_1000"],
    "max_depth (xgboost)": ["xgb_depth_2", "xgb_baseline", "xgb_depth_8"],
    "n_estimators (xgboost)": ["xgb_n_est_50", "xgb_baseline", "xgb_n_est_600"],
    "learning_rate (xgboost)": ["xgb_lr_0.02", "xgb_baseline", "xgb_lr_0.3"],
    "semilla de partición (ridge)": ["ridge_baseline", "ridge_seed_7"],
}


def cargar_corridas(tracking_uri: str, experimento: str) -> pd.DataFrame:
    """Lee las corridas terminadas del experimento, ordenadas por cv_mae."""
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experimento)
    if exp is None:
        raise SystemExit(f"No existe el experimento '{experimento}' en {tracking_uri}")
    corridas = mlflow.search_runs(
        experiment_ids=[exp.experiment_id], order_by=["metrics.cv_mae ASC"]
    )
    return corridas[corridas["status"] == "FINISHED"].reset_index(drop=True)


def tabla_resumen(corridas: pd.DataFrame) -> pd.DataFrame:
    """Una fila por corrida con los parámetros y métricas de COLUMNAS."""
    return corridas[list(COLUMNAS)].rename(columns=COLUMNAS)


def tabla_influencia(corridas: pd.DataFrame) -> pd.DataFrame:
    """Rango de cv_mae y del MAE de prueba dentro de cada grupo de GRUPOS_INFLUENCIA."""
    indexadas = corridas.set_index("tags.mlflow.runName")
    if indexadas.index.duplicated().any():
        raise SystemExit("Hay nombres de corrida repetidos; no se puede calcular la influencia.")
    filas = []
    for factor, nombres in GRUPOS_INFLUENCIA.items():
        grupo = indexadas.loc[nombres]
        cv = grupo["metrics.cv_mae"]
        prueba = grupo["metrics.mae"]
        filas.append(
            {
                "factor": factor,
                "corridas": ", ".join(nombres),
                "cv_mae_minimo": cv.min(),
                "cv_mae_maximo": cv.max(),
                "rango_cv_mae": cv.max() - cv.min(),
                "rango_mae_prueba": prueba.max() - prueba.min(),
            }
        )
    return pd.DataFrame(filas).sort_values("rango_cv_mae", ascending=False).reset_index(drop=True)


def formatear(valor, decimales: int = 4) -> str:
    """Texto de una celda: números con decimales fijos, vacío si es nulo."""
    if pd.isna(valor):
        return ""
    if isinstance(valor, float):
        return f"{valor:.{decimales}f}"
    return str(valor)


def a_markdown(tabla: pd.DataFrame) -> str:
    """Convierte un DataFrame en una tabla Markdown."""
    encabezado = "| " + " | ".join(tabla.columns) + " |"
    separador = "|" + "|".join("---" for _ in tabla.columns) + "|"
    filas = [
        "| " + " | ".join(formatear(valor) for valor in fila) + " |"
        for _, fila in tabla.iterrows()
    ]
    return "\n".join([encabezado, separador, *filas])


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta el resumen de corridas de MLflow a results/.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI, help="URI de MLflow Tracking.")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT, help="Experimento a exportar.")
    parser.add_argument("--out-dir", type=Path, default=REPO_DIR / "results", help="Carpeta de salida.")
    args = parser.parse_args()

    corridas = cargar_corridas(args.tracking_uri, args.experiment_name)
    resumen = tabla_resumen(corridas)
    influencia = tabla_influencia(corridas)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(args.out_dir / "runs_summary.csv", index=False)
    influencia.to_csv(args.out_dir / "influencia_hiperparametros.csv", index=False)
    texto = (
        f"# Resumen de corridas: {args.experiment_name}\n\n"
        f"Corridas terminadas: {len(resumen)}. Ordenadas por cv_mae (menor es mejor).\n\n"
        f"{a_markdown(resumen)}\n\n"
        "## Influencia de cada factor (un factor a la vez)\n\n"
        f"{a_markdown(influencia)}\n"
    )
    (args.out_dir / "runs_summary.md").write_text(texto, encoding="utf-8")
    print(f"{len(resumen)} corridas exportadas a {args.out_dir}")


if __name__ == "__main__":
    main()
