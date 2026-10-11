"""Entrena el modelo base de fatiga del día siguiente y registra la corrida en MLflow.

Es la migración de las secciones 3 a 6 del notebook de la Actividad 2 a un script
con argumentos de línea de comandos. Sin argumentos entrena el Ridge baseline, que
reproduce las métricas del notebook (MAE de prueba 1.398).

Ejemplos:
    python train.py
    python train.py --model xgboost --max-depth 2 --run-name xgb_depth_2
    python train.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

# Evita el aviso informativo del agente de MLflow que aparece al importar la librería.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

REPO_DIR = Path(__file__).resolve().parent
DATASET_KAGGLE = "samartalwar/sleep-debt-and-screen-time-late-night-phone-habits"
CSV_NAME = "bedtime_screentime_sleep_debt.csv"
OBJETIVO = "next_day_fatigue_score"
FEATURES_NUM = [
    "bedtime_phone_minutes",
    "screen_brightness_pct",
    "blue_light_filter_active",
    "caffeine_post_5pm_mg",
]
FEATURES_CAT = ["primary_bedtime_app"]
FEATURES = FEATURES_NUM + FEATURES_CAT
# Rangos documentados del dataset: (mínimo, máximo). None significa sin máximo.
RANGOS = {
    OBJETIVO: (1, 10),
    "screen_brightness_pct": (10, 100),
    "blue_light_filter_active": (0, 1),
    "bedtime_phone_minutes": (0, None),
    "caffeine_post_5pm_mg": (0, None),
}
# Hiperparámetros de XGBoost que se fijan en el código (no son argumentos).
XGB_SUBSAMPLE = 0.9
XGB_COLSAMPLE = 0.9
# skops exige declarar estos tipos de XGBoost al guardar el modelo; el archivo lo generamos nosotros.
TIPOS_CONFIABLES_XGBOOST = ["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"]
DEFAULT_EXPERIMENT = "tc5061_act3_fatiga_dia_siguiente"
DEFAULT_DB = REPO_DIR / "mlflow.db"
DEFAULT_TRACKING_URI = f"sqlite:///{DEFAULT_DB}"
NOTA_CORRIDA = (
    "Corrida de la Actividad 3 (TC5061). Modelo base de fatiga del día siguiente. "
    "Los hiperparámetros de esta corrida están registrados como parámetros."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lee los argumentos de línea de comandos. Todos tienen valor por defecto."""
    parser = argparse.ArgumentParser(
        description="Entrena el modelo base de fatiga del día siguiente y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["ridge", "xgboost"],
        default="ridge",
        help="Algoritmo a entrenar.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla de la partición train/test y de XGBoost.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fracción del dataset reservada para prueba. Debe estar entre 0 y 1.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Pliegues de validación cruzada sobre el conjunto de entrenamiento.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="[ridge] Fuerza de la penalización L2.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="[xgboost] Número de árboles.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="[xgboost] Profundidad máxima de cada árbol.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.08,
        help="[xgboost] Tasa de aprendizaje (eta).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="CSV local del dataset. Si no se indica, se descarga con kagglehub (usa caché).",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT,
        help="Nombre del experimento en MLflow.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre de la corrida. Si no se indica, se genera con los hiperparámetros.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="URI de MLflow Tracking. Si no se indica, usa mlflow.db dentro de la carpeta del repo.",
    )
    args = parser.parse_args(argv)
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1, sin incluir los extremos.")
    if args.cv_folds < 2:
        parser.error("--cv-folds debe ser al menos 2.")
    return args


def set_seed(seed: int) -> None:
    """Fija las semillas globales de Python y NumPy. Las de XGBoost y la partición van por argumento."""
    random.seed(seed)
    np.random.seed(seed)


def load_data(data_path: Path | None) -> tuple[pd.DataFrame, Path]:
    """Lee el CSV local o lo descarga con kagglehub (sección 1 del notebook)."""
    if data_path is None:
        import kagglehub  # importación diferida: --help no necesita kagglehub

        data_path = Path(kagglehub.dataset_download(DATASET_KAGGLE)) / CSV_NAME
    return pd.read_csv(data_path), data_path


def validate_data(df: pd.DataFrame) -> None:
    """Revisa nulos, filas duplicadas, user_id repetidos y rangos documentados (sección 3).

    Detiene el entrenamiento si algo falla. El notebook solo lo reportaba.
    """
    problemas = []
    nulos = int(df.isna().sum().sum())
    if nulos:
        problemas.append(f"{nulos} valores nulos")
    duplicados = int(df.duplicated().sum())
    if duplicados:
        problemas.append(f"{duplicados} filas duplicadas")
    if "user_id" in df.columns and df["user_id"].duplicated().any():
        problemas.append("user_id repetidos")
    for columna, (minimo, maximo) in RANGOS.items():
        fuera = df[columna] < minimo
        if maximo is not None:
            fuera = fuera | (df[columna] > maximo)
        if fuera.any():
            problemas.append(f"{int(fuera.sum())} valores fuera de rango en {columna}")
    if problemas:
        raise ValueError("La validación de calidad falló: " + "; ".join(problemas))


def build_model_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Entradas y salida del modelo. El resto de columnas queda fuera a propósito (ver notebook, sección 2)."""
    tabla = df[FEATURES + [OBJETIVO]].copy()
    tabla[FEATURES_NUM] = tabla[FEATURES_NUM].astype("float64")
    return tabla[FEATURES], tabla[OBJETIVO]


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    """Partición train/test. Se hace antes de ajustar cualquier preprocesamiento para no filtrar datos de prueba."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Preprocesamiento y modelo en un solo Pipeline, para que la validación cruzada no filtre información.

    Ridge necesita las variables numéricas escaladas; XGBoost no (passthrough).
    La app principal se codifica con one-hot en ambos casos.
    """
    es_ridge = args.model == "ridge"
    preprocesamiento = ColumnTransformer(
        [
            ("num", StandardScaler() if es_ridge else "passthrough", FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ]
    )
    if es_ridge:
        modelo = Ridge(alpha=args.alpha)
    else:
        modelo = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE,
            objective="reg:squarederror",
            random_state=args.seed,
            n_jobs=1,
        )
    return Pipeline([("pre", preprocesamiento), ("model", modelo)])


def cross_validate_mae(estimador, X_train, y_train, cv_folds: int) -> tuple[float, float]:
    """MAE medio y desviación estándar en validación cruzada, solo sobre el entrenamiento.

    Con cv entero, scikit-learn usa KFold sin barajar: el resultado no depende de la semilla.
    """
    errores = -cross_val_score(
        estimador, X_train, y_train, cv=cv_folds, scoring="neg_mean_absolute_error"
    )
    return float(errores.mean()), float(errores.std())


def evaluate(pipeline: Pipeline, X_train, X_test, y_train, y_test):
    """Métricas en prueba y en entrenamiento, y el MAE de predecir siempre la media del entrenamiento."""
    pred = pipeline.predict(X_test)
    media = np.full(len(y_test), y_train.mean())
    metricas = {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "mape": mean_absolute_percentage_error(y_test, pred),
        "r2": r2_score(y_test, pred),
        "train_mae": mean_absolute_error(y_train, pipeline.predict(X_train)),
        "mae_media": mean_absolute_error(y_test, media),
    }
    return {clave: float(valor) for clave, valor in metricas.items()}, pred


def feature_table(pipeline: Pipeline, modelo: str) -> pd.DataFrame:
    """Coeficientes de Ridge (estandarizados y en unidades originales) o importancias de XGBoost."""
    preprocesamiento = pipeline.named_steps["pre"]
    nombres_cat = list(preprocesamiento.named_transformers_["cat"].get_feature_names_out(FEATURES_CAT))
    nombres = FEATURES_NUM + nombres_cat
    estimador = pipeline.named_steps["model"]
    if modelo == "ridge":
        escala = np.concatenate(
            [preprocesamiento.named_transformers_["num"].scale_, np.ones(len(nombres_cat))]
        )
        return pd.DataFrame(
            {
                "feature": nombres,
                "coef_estandarizado": estimador.coef_,
                "coef_unidad_original": estimador.coef_ / escala,
            }
        )
    return pd.DataFrame(
        {"feature": nombres, "importancia_gain": estimador.feature_importances_}
    )


def make_eval_figure(y_test: pd.Series, pred: np.ndarray, ruta: Path) -> None:
    """Figura de evaluación: real contra predicho y distribución de residuos (sección 5)."""
    residuo = y_test.to_numpy() - pred
    figura = Figure(figsize=(10, 4))
    ax_dispersion, ax_residuos = figura.subplots(1, 2)
    ax_dispersion.scatter(y_test, pred, alpha=0.35, s=16, color="#4C78A8", edgecolors="none")
    ax_dispersion.plot([0, 11], [0, 11], color="#E45756", linestyle="--", label="predicción perfecta")
    ax_dispersion.set(
        xlim=(0, 11),
        ylim=(0, 11),
        xlabel="Fatiga real",
        ylabel="Fatiga predicha",
        title="Prueba: real contra predicho",
    )
    ax_dispersion.legend()
    ax_residuos.hist(residuo, bins=30, color="#72B7B2")
    ax_residuos.axvline(0, color="#E45756", linestyle="--")
    ax_residuos.set(xlabel="Real − predicho", title="Residuos en prueba")
    figura.tight_layout()
    figura.savefig(ruta, dpi=120)


def sha256_archivo(ruta: Path) -> str:
    """Huella SHA-256 del archivo de datos, para saber con qué datos se entrenó."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def ruta_sqlite(tracking_uri: str) -> Path | None:
    """Ruta del archivo SQLite si la URI apunta a una base local; si no, None."""
    prefijo = "sqlite:///"
    if tracking_uri.startswith(prefijo):
        return Path(tracking_uri[len(prefijo):])
    return None


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Conecta con la base de MLflow y crea el experimento si no existe.

    Con SQLite local, los artefactos se guardan en mlartifacts/<experimento>/ junto a la base.
    Así la interfaz encuentra los archivos sin importar desde qué carpeta se lance.
    """
    base = ruta_sqlite(tracking_uri)
    if base is not None:
        base.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        opciones = {}
        if base is not None:
            carpeta = base.resolve().parent / "mlartifacts" / experiment_name
            carpeta.mkdir(parents=True, exist_ok=True)
            opciones["artifact_location"] = carpeta.as_uri()
        mlflow.create_experiment(experiment_name, **opciones)
    mlflow.set_experiment(experiment_name)


def nombre_por_defecto(args: argparse.Namespace) -> str:
    """Nombre de la corrida cuando no se pasa --run-name."""
    if args.model == "ridge":
        return f"ridge_alpha_{args.alpha}"
    return f"xgb_n{args.n_estimators}_d{args.max_depth}_lr{args.learning_rate}"


def log_to_mlflow(
    args: argparse.Namespace,
    pipeline: Pipeline,
    metricas: dict[str, float],
    y_test: pd.Series,
    pred: np.ndarray,
    X_train: pd.DataFrame,
    info_datos: dict,
) -> None:
    """Registra en la corrida activa: hiperparámetros, métricas, artefactos y el modelo."""
    parametros = {
        "modelo": args.model,
        "seed": args.seed,
        "test_size": args.test_size,
        "cv_folds": args.cv_folds,
        "objetivo": OBJETIVO,
        "features": ",".join(FEATURES),
        "metrica_principal": "mae",
        **info_datos,
    }
    if args.model == "ridge":
        parametros["alpha"] = args.alpha
    else:
        parametros.update(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE,
        )
    for clave, valor in parametros.items():
        mlflow.log_param(clave, valor)
    for clave, valor in metricas.items():
        mlflow.log_metric(clave, valor)
    mlflow.set_tag("familia_modelo", "lineal" if args.model == "ridge" else "boosting_arboles")
    mlflow.set_tag("mlflow.note.content", NOTA_CORRIDA)

    with tempfile.TemporaryDirectory() as tmp:
        carpeta = Path(tmp)
        figura = carpeta / "evaluacion_prueba.png"
        make_eval_figure(y_test, pred, figura)
        metricas_json = carpeta / "metricas.json"
        metricas_json.write_text(json.dumps(metricas, indent=2), encoding="utf-8")
        tabla = feature_table(pipeline, args.model)
        ruta_tabla = carpeta / ("coeficientes.csv" if args.model == "ridge" else "importancias.csv")
        tabla.to_csv(ruta_tabla, index=False)
        for ruta in (figura, metricas_json, ruta_tabla):
            mlflow.log_artifact(str(ruta))

    firma = infer_signature(X_train, pipeline.predict(X_train.head(5)))
    mlflow.sklearn.log_model(
        pipeline,
        name="modelo",
        signature=firma,
        input_example=X_train.head(2),
        skops_trusted_types=TIPOS_CONFIABLES_XGBOOST if args.model == "xgboost" else None,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    df, ruta_csv = load_data(args.data_path)
    validate_data(df)
    X, y = build_model_table(df)
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)
    info_datos = {
        "dataset": ruta_csv.name,
        "data_sha256": sha256_archivo(ruta_csv),
        "n_filas": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    configure_mlflow(args.tracking_uri or DEFAULT_TRACKING_URI, args.experiment_name)
    nombre = args.run_name or nombre_por_defecto(args)
    with mlflow.start_run(run_name=nombre) as run:
        pipeline = build_pipeline(args)
        cv_mae, cv_mae_std = cross_validate_mae(pipeline, X_train, y_train, args.cv_folds)
        cv_mae_media, _ = cross_validate_mae(
            DummyRegressor(strategy="mean"), X_train, y_train, args.cv_folds
        )
        pipeline.fit(X_train, y_train)
        metricas, pred = evaluate(pipeline, X_train, X_test, y_train, y_test)
        metricas.update(cv_mae=cv_mae, cv_mae_std=cv_mae_std, cv_mae_media=cv_mae_media)
        log_to_mlflow(args, pipeline, metricas, y_test, pred, X_train, info_datos)
        run_id = run.info.run_id

    print(f"Experimento: {args.experiment_name}")
    print(f"Corrida: {nombre} ({run_id})")
    print(
        f"MAE prueba: {metricas['mae']:.4f} | CV MAE: {cv_mae:.4f} "
        f"| R2 prueba: {metricas['r2']:.4f}"
    )
    print(f"Tracking: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
