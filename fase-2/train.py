"""Entrenamiento del modelo de riesgo de diabetes (Fase 2).

Uso (desde la carpeta fase-2, en una sola línea):

    python train.py --data data/datos.csv --target Diabetes_Risk \
        --output-dir models --version v1 --seed 42

Qué hace, de principio a fin:
  1. Lee el CSV y valida que existan la variable objetivo y las columnas.
  2. Separa entrenamiento/prueba (estratificado) ANTES de ajustar transformaciones.
  3. Compara configuraciones (class_weight=None vs 'balanced') con validación
     cruzada SOLO sobre el conjunto de entrenamiento. El preprocesamiento va
     dentro del Pipeline, así que se reajusta en cada fold (sin fuga).
  4. Reentrena la mejor configuración con todo el entrenamiento y la evalúa UNA
     vez sobre prueba, junto con el DummyClassifier (misma partición y métrica).
  5. Guarda en <output-dir>/<version>/: modelo.joblib (pipeline completo),
     metrics.json y metadata.json. Nunca sobrescribe una versión existente.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_TARGET = "Diabetes_Risk"
# Columnas que NO deben entrar al modelo: identificador y variables derivadas del riesgo
# (la calificación de la Fase 1 las excluye del pipeline final).
DEFAULT_DROP_COLS = ["Patient_ID", "Diabetes_Risk_Score", "Doctor_Consultation_Needed"]
# Además se excluye automáticamente cualquier columna cuyo nombre sugiera una
# recomendación derivada del riesgo (p. ej. AI_Health_Recommendation).
LEAKAGE_NAME_PATTERN = re.compile(r"recomend|recommend", re.IGNORECASE)
CLASS_WEIGHT_CANDIDATES = [None, "balanced"]
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrainingError(Exception):
    """Error previsible de entrenamiento (datos, argumentos o versión repetida)."""


# --------------------------------------------------------------------------- #
# Argumentos
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entrena el pipeline (preprocesamiento + Random Forest), lo compara "
        "con un DummyClassifier y guarda una versión identificable del modelo."
    )
    p.add_argument("--data", required=True, help="Ruta del CSV con datos etiquetados.")
    p.add_argument("--target", default=DEFAULT_TARGET,
                   help=f"Nombre de la variable objetivo (por defecto: {DEFAULT_TARGET}).")
    p.add_argument("--output-dir", required=True,
                   help="Directorio donde se crean las carpetas de versión (ej. models).")
    p.add_argument("--version", required=True,
                   help="Identificador único de la versión (ej. v1). No se sobrescribe.")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (por defecto: 42).")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Proporción de prueba, entre 0 y 1 (por defecto: 0.2).")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Folds de validación cruzada sobre el entrenamiento (por defecto: 5).")
    p.add_argument("--n-estimators", type=int, default=200,
                   help="Árboles del Random Forest (por defecto: 200).")
    p.add_argument("--min-samples-leaf", type=int, default=5,
                   help="Mínimo de muestras por hoja del Random Forest; reduce el tamaño del "
                        "modelo (por defecto: 5).")
    p.add_argument("--drop-cols", nargs="*", default=None,
                   help="Columnas a excluir del modelo (por defecto: "
                        f"{' '.join(DEFAULT_DROP_COLS)} + columnas de recomendaciones).")
    p.add_argument("--n-jobs", type=int, default=-1, help="Procesos en paralelo (por defecto: -1).")
    return p


def validate_args(args: argparse.Namespace) -> None:
    if not VERSION_PATTERN.match(args.version):
        raise TrainingError(
            f"Versión inválida '{args.version}': use letras, números, '.', '_' o '-' (ej. v1)."
        )
    if not 0 < args.test_size < 1:
        raise TrainingError("--test-size debe estar entre 0 y 1 (exclusivo).")
    if args.cv_folds < 2:
        raise TrainingError("--cv-folds debe ser al menos 2.")
    if args.n_estimators < 1:
        raise TrainingError("--n-estimators debe ser al menos 1.")


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def load_data(path: str | Path, target: str, drop_cols: list[str] | None = None):
    """Lee el CSV, valida el objetivo y devuelve (X, y, columnas_excluidas)."""
    path = Path(path)
    if not path.is_file():
        raise TrainingError(f"No existe el archivo de datos: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # CSV ilegible
        raise TrainingError(f"No se pudo leer el CSV '{path}': {exc}") from exc
    if target not in df.columns:
        raise TrainingError(
            f"La variable objetivo '{target}' no está en el CSV. Columnas: {list(df.columns)}"
        )
    if df[target].isna().any():
        raise TrainingError(f"La variable objetivo '{target}' tiene valores nulos.")

    candidates = list(DEFAULT_DROP_COLS if drop_cols is None else drop_cols)
    candidates += [c for c in df.columns if LEAKAGE_NAME_PATTERN.search(c)]
    dropped = [c for c in dict.fromkeys(candidates) if c in df.columns and c != target]

    y = df[target]
    X = df.drop(columns=[target] + dropped)
    if X.shape[1] == 0:
        raise TrainingError("No quedan variables predictoras después de excluir columnas.")
    if y.value_counts().min() < 2:
        raise TrainingError("Alguna clase tiene menos de 2 registros; no se puede estratificar.")
    return X, y, dropped


def split_column_types(X: pd.DataFrame):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Imputación + escalamiento (numéricas) e imputación + one-hot (categóricas).

    Las categorías no vistas en entrenamiento se ignoran (todo ceros) al predecir.
    """
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer(
        [("num", num, numeric), ("cat", cat, categorical)], remainder="drop"
    )


def build_pipeline(numeric, categorical, seed, n_estimators, class_weight, n_jobs,
                   min_samples_leaf=5) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators, class_weight=class_weight,
            min_samples_leaf=min_samples_leaf,
            random_state=seed, n_jobs=n_jobs)),
    ])


def evaluate(estimator, X, y, labels) -> dict:
    pred = estimator.predict(X)
    report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "per_class": {str(l): {k: float(v) for k, v in report[str(l)].items()} for l in labels},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "confusion_matrix_labels": [str(l) for l in labels],
    }


def cross_validate_configs(X_train, y_train, numeric, categorical, labels, args) -> dict:
    """Compara class_weight con CV estratificada SOLO sobre entrenamiento.

    El preprocesamiento está dentro del Pipeline, por lo que se ajusta en cada fold.
    """
    scoring = {"f1_macro": "f1_macro", "accuracy": "accuracy"}
    for label in labels:  # recall por clase (evidencia sobre la clase minoritaria)
        scoring[f"recall_{label}"] = make_scorer(
            recall_score, labels=[label], average="macro", zero_division=0)

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    results = {}
    for cw in CLASS_WEIGHT_CANDIDATES:
        pipe = build_pipeline(numeric, categorical, args.seed, args.n_estimators, cw, args.n_jobs,
                              args.min_samples_leaf)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        name = "none" if cw is None else cw
        results[name] = {
            "class_weight": cw,
            **{m: {"mean": float(np.mean(scores[f"test_{m}"])),
                   "std": float(np.std(scores[f"test_{m}"]))} for m in scoring},
        }
    return results


# --------------------------------------------------------------------------- #
# Versionado y metadatos
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def check_version_free(output_dir: Path, version: str) -> Path:
    version_dir = output_dir / version
    if version_dir.exists():
        raise TrainingError(
            f"La versión '{version}' ya existe en {output_dir}. No se sobrescribe: "
            "use otro identificador con --version."
        )
    return version_dir


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Flujo principal
# --------------------------------------------------------------------------- #
def run_training(args: argparse.Namespace) -> Path:
    """Ejecuta el entrenamiento completo y devuelve la carpeta de la versión creada."""
    validate_args(args)
    output_dir = Path(args.output_dir)
    check_version_free(output_dir, args.version)  # falla rápido, antes de entrenar

    X, y, dropped = load_data(args.data, args.target, args.drop_cols)
    numeric, categorical = split_column_types(X)
    labels = sorted(y.unique())

    # 1) Partición antes de ajustar cualquier transformación.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    # 2) Selección de configuración con CV dentro del entrenamiento.
    cv_results = cross_validate_configs(X_train, y_train, numeric, categorical, labels, args)
    best_name = max(cv_results, key=lambda k: cv_results[k]["f1_macro"]["mean"])
    best_cw = cv_results[best_name]["class_weight"]

    # 3) Reentrenamiento y evaluación final (prueba se usa una sola vez).
    pipeline = build_pipeline(numeric, categorical, args.seed, args.n_estimators, best_cw, args.n_jobs,
                              args.min_samples_leaf)
    pipeline.fit(X_train, y_train)
    model_metrics = evaluate(pipeline, X_test, y_test, labels)

    baseline = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(baseline, X_test, y_test, labels)

    metrics = {
        "metric": "f1_macro",
        "test": {"model": model_metrics, "baseline_dummy_most_frequent": baseline_metrics},
        "f1_macro_gain_vs_baseline": model_metrics["f1_macro"] - baseline_metrics["f1_macro"],
        "cross_validation": {
            "scope": "solo conjunto de entrenamiento; preprocesamiento dentro del Pipeline por fold",
            "folds": args.cv_folds,
            "selection_criterion": "f1_macro (media)",
            "selected_config": best_name,
            "results": cv_results,
        },
    }

    data_path = Path(args.data)
    metadata = {
        "version": args.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "problem_type": "clasificación multiclase",
        "target": args.target,
        "classes": [str(l) for l in labels],
        "algorithm": "RandomForestClassifier",
        "pipeline_steps": ["ColumnTransformer(imputación + StandardScaler + OneHotEncoder)",
                           "RandomForestClassifier"],
        "model_params": {"n_estimators": args.n_estimators, "class_weight": best_cw,
                         "min_samples_leaf": args.min_samples_leaf, "random_state": args.seed},
        "seed": args.seed,
        "missing_values_strategy": {"numeric": "mediana (SimpleImputer)",
                                    "categorical": "moda (SimpleImputer most_frequent)",
                                    "unseen_categories": "ignoradas por OneHotEncoder (handle_unknown='ignore')"},
        "input_features": {
            "numeric": numeric, "categorical": categorical,
            "dtypes": {c: str(X[c].dtype) for c in X.columns},
        },
        "excluded_columns": dropped,
        "data": {"file": data_path.name, "sha256": sha256_file(data_path),
                 "n_rows": int(len(X)), "n_features": int(X.shape[1])},
        "split": {"strategy": "train_test_split estratificado", "test_size": args.test_size,
                  "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                  "random_state": args.seed},
        "evaluation_metric": "f1_macro",
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "requirements_file": "requirements.txt",
        "git_commit": git_commit(),
        "reproducibility_note": "Random Forest es estocástico; con la misma semilla, datos y "
                                "versiones de librerías los resultados son consistentes.",
    }

    # 4) Guardado: la carpeta se crea al final (exist_ok=False evita sobrescribir).
    version_dir = output_dir / args.version
    try:
        version_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise TrainingError(f"La versión '{args.version}' ya existe; no se sobrescribe.") from exc
    joblib.dump(pipeline, version_dir / "modelo.joblib", compress=3)
    write_json(version_dir / "metrics.json", metrics)
    write_json(version_dir / "metadata.json", metadata)

    print_summary(args, best_name, cv_results, model_metrics, baseline_metrics, version_dir)
    return version_dir


def print_summary(args, best_name, cv_results, model_m, base_m, version_dir) -> None:
    print(f"\n=== Entrenamiento {args.version} (seed={args.seed}) ===")
    print("Validación cruzada (solo entrenamiento), F1-macro:")
    for name, r in cv_results.items():
        mark = "  <- seleccionado" if name == best_name else ""
        print(f"  class_weight={name:<9} {r['f1_macro']['mean']:.4f} ± {r['f1_macro']['std']:.4f}{mark}")
    print("Evaluación final en prueba:")
    print(f"  Modelo   F1-macro={model_m['f1_macro']:.4f}  accuracy={model_m['accuracy']:.4f}")
    print(f"  Baseline F1-macro={base_m['f1_macro']:.4f}  accuracy={base_m['accuracy']:.4f}")
    print("Archivos generados:")
    for name in ("modelo.joblib", "metrics.json", "metadata.json"):
        print(f"  {version_dir / name}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_training(args)
    except TrainingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
