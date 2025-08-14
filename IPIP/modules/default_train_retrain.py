# pipeline/modules/default_train_retrain.py
# -*- coding: utf-8 -*-
"""
Entrenamiento y reentrenamiento por defecto — versión IPIP

- IPIPClassifier: ensemble de ensembles con muestreo balanceado por clase,
  construcción incremental de cada ensemble con criterio de mejora de Balanced Accuracy.
- default_train: entrena IPIP con split holdout.
- default_retrain: reentrena IPIP según distintos modos (0=full, 2=ventana, 5=replay mix, 6=recalib -> full).
"""
import os
import json
import math
import copy
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, classification_report
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# =========================================================
# Utilidades IPIP
# =========================================================

def _safe_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def _balanced_sample_indices(y: np.ndarray, n_pos_target: int, prop_majoritaria: float, rng: np.random.RandomState):
    """
    Genera índices para un dataset balanceado:
      - Toma n_pos_target positivos con reposición
      - Toma n_neg_target tal que % mayoritaria ≈ prop_majoritaria (e.g., 0.55)
    """
    y = np.asarray(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        # Degenerado: devuelve todo lo que haya
        return np.arange(len(y))

    n_pos = max(1, int(n_pos_target))
    # prop_majoritaria = n_neg / (n_pos + n_neg)
    # => n_neg = prop_majoritaria * (n_pos + n_neg) => n_neg * (1 - prop) = prop * n_pos
    # => n_neg = prop * n_pos / (1 - prop)
    n_neg = int(round(prop_majoritaria * n_pos / max(1e-9, (1.0 - prop_majoritaria))))
    # sample with replacement
    sel_pos = rng.choice(pos_idx, size=n_pos, replace=True)
    sel_neg = rng.choice(neg_idx, size=max(1, n_neg), replace=True)
    return np.concatenate([sel_pos, sel_neg])

def _majority_vote(labels_2d: np.ndarray):
    """
    Recibe matriz shape (n_models, n_samples) de etiquetas (enteras) y retorna vector de votos mayoritarios.
    """
    # mode por columna
    from scipy.stats import mode
    m = mode(labels_2d, axis=0, keepdims=False)
    return m.mode

def _mean_probs(probs_list: list):
    """
    Recibe lista de arrays [n_samples, n_classes] y devuelve media elemento a elemento.
    """
    return np.nanmean(np.stack(probs_list, axis=0), axis=0)

def _safe_balanced_accuracy(y_true, y_pred):
    try:
        return float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        return 0.0

def _compute_p_b(n_pos: int, eps: float = 0.01) -> Tuple[int, int]:
    """
    Replica las fórmulas del script R (con defensas) para p y b.
    np = round(n_pos * 0.75)
    p  = ceil( log(.01) / ( log(1 - 1/n_pos) * np ) )
    b  = ceil( log(.01) / ( log(1 - 1/np)   * np ) )
    """
    n_pos = max(2, int(n_pos))
    np_target = max(1, int(round(n_pos * 0.75)))
    try:
        p = int(math.ceil(math.log(eps) / (math.log(1 - 1.0 / n_pos) * np_target)))
    except Exception:
        p = 3
    try:
        b = int(math.ceil(math.log(eps) / (math.log(1 - 1.0 / max(1, np_target)) * np_target)))
    except Exception:
        b = 3
    return max(1, p), max(1, b), np_target

def _patience(len_ensemble: int) -> int:
    """Número de intentos sin mejora permitidos al ampliar un ensemble (mt en R)."""
    # Puedes sofisticarlo en función de len_ensemble. De momento fijo=3.
    return 3

# =========================================================
# IPIPClassifier
# =========================================================
class IPIPClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble de ensembles:
      - p ensembles (E_k), cada uno se construye incrementalmente añadiendo modelos base
        si y solo si mejora la Balanced Accuracy en un conjunto de validación.
      - Cada modelo base se entrena sobre una muestra balanceada (bootstrap) usando prop_majoritaria.
      - Predicción: voto mayoritario (predict) y media de probabilidades (predict_proba).

    Parametros:
      base_estimator: estimador sklearn para clonar (default: RandomForestClassifier)
      prop_majoritaria: fracción objetivo de la clase mayoritaria en la muestra balanceada (0.55 por defecto)
      random_state: RNG
      val_size: tamaño de validación interna (0.2)
      p, b: si None, se calculan a partir de la cantidad de positivos
    """
    def __init__(
        self,
        base_estimator=None,
        prop_majoritaria: float = 0.55,
        random_state: int = 42,
        val_size: float = 0.2,
        p: Optional[int] = None,
        b: Optional[int] = None
    ):
        self.base_estimator = base_estimator if base_estimator is not None else RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=random_state
        )
        self.prop_majoritaria = float(prop_majoritaria)
        self.random_state = int(random_state)
        self.val_size = float(val_size)
        self.p = p
        self.b = b

        # Atributos tras fit
        self.ensembles_ = None           # List[List[estimator]]
        self.classes_ = None
        self._pos_label_ = None          # valor de clase positiva (para proba index)
        self.meta_ = {}

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)

        # Normalizamos clases a {0,1,...}, pero mantenemos mapping natural
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            # IPIP está pensado para binario; si no, mapeamos a {0,1}
            # Tomamos la clase más frecuente como 0 y la otra como 1
            vals, counts = np.unique(y, return_counts=True)
            order = np.argsort(-counts)
            mapping = {vals[order[0]]: 0, vals[order[1]]: 1}
            y_bin = np.array([mapping[v] if v in mapping else 1 for v in y], dtype=int)
            self.classes_ = np.array([0,1], dtype=int)
        else:
            # Map al orden de aparicion: tomamos la primera como 0, segunda 1
            mapping = {self.classes_[0]: 0, self.classes_[1]: 1}
            y_bin = np.array([mapping[v] for v in y], dtype=int)

        self._pos_label_ = 1

        # Split validación interna para "mejora" del ensemble
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_bin, test_size=self.val_size, random_state=self.random_state, stratify=y_bin
        )

        n_pos = int(np.sum(y_tr == 1))
        p, b, np_target = _compute_p_b(n_pos)
        if self.p is not None: p = int(self.p)
        if self.b is not None: b = int(self.b)

        ensembles = []
        # Prepara dataset auxiliar balanceado por ensemble (como en R creaban dfs[[k]])
        for k in range(p):
            Ek = []
            patience_left = _patience(len(Ek))
            # Balanced starting indices
            idx_bal_start = _balanced_sample_indices(y_tr, n_pos_target=np_target, prop_majoritaria=self.prop_majoritaria, rng=rng)
            X_bal = X_tr[idx_bal_start]
            y_bal = y_tr[idx_bal_start]

            # Función para evaluar BA del ensemble en val
            def eval_ensemble(models_list):
                if len(models_list) == 0:
                    return -np.inf
                # Voto mayoritario de los modelos ya añadidos
                preds_each = []
                for m in models_list:
                    preds_each.append(m.predict(X_val))
                preds_each = np.stack(preds_each, axis=0)    # (n_models, n_samples)
                y_pred = _majority_vote(preds_each)
                return _safe_balanced_accuracy(y_val, y_pred)

            # Métrica del ensemble actual (vacío)
            best_ba = eval_ensemble(Ek)

            # Construcción incremental
            while len(Ek) < b and patience_left > 0:
                # Entrena un modelo base sobre una muestra balanceada (re-sampling sobre X_bal,y_bal)
                idx_bal = _balanced_sample_indices(y_bal, n_pos_target=np_target, prop_majoritaria=self.prop_majoritaria, rng=rng)
                Xb = X_bal[idx_bal]
                yb = y_bal[idx_bal]

                m = clone(self.base_estimator)
                if hasattr(m, "random_state"):
                    m.set_params(random_state=rng.randint(0, 2**31 - 1))
                m.fit(Xb, yb)

                # Evalúa si mejora el ensemble
                Ek_candidate = Ek + [m]
                ba2 = eval_ensemble(Ek_candidate)

                if ba2 > best_ba:
                    Ek = Ek_candidate
                    best_ba = ba2
                    patience_left = _patience(len(Ek))  # reset paciencia tras mejora
                else:
                    patience_left -= 1

            # Si por algún motivo no se añadió ningún modelo, añade al menos uno
            if len(Ek) == 0:
                m = clone(self.base_estimator)
                if hasattr(m, "random_state"):
                    m.set_params(random_state=rng.randint(0, 2**31 - 1))
                m.fit(X_bal, y_bal)
                Ek = [m]

            ensembles.append(Ek)

        self.ensembles_ = ensembles
        self.meta_ = {
            "p": p,
            "b": b,
            "np_target": np_target,
            "val_size": self.val_size,
            "prop_majoritaria": self.prop_majoritaria
        }
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.ensembles_ is None or len(self.ensembles_) == 0:
            raise RuntimeError("IPIPClassifier not fitted.")
        # voto mayoritario a dos niveles: por ensemble y después entre ensembles
        votes_ensembles = []
        for Ek in self.ensembles_:
            votes_models = []
            for m in Ek:
                votes_models.append(m.predict(X))
            votes_models = np.stack(votes_models, axis=0)  # (n_models, n_samples)
            votes_Ek = _majority_vote(votes_models)        # (n_samples,)
            votes_ensembles.append(votes_Ek)
        votes_ensembles = np.stack(votes_ensembles, axis=0)
        final_votes = _majority_vote(votes_ensembles)      # (n_samples,)
        # map back to original classes if needed
        # Aquí mantenemos 0/1 (binario). Devuelve 0/1.
        return final_votes

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.ensembles_ is None or len(self.ensembles_) == 0:
            raise RuntimeError("IPIPClassifier not fitted.")
        probs_ensembles = []
        for Ek in self.ensembles_:
            probs_models = []
            for m in Ek:
                if hasattr(m, "predict_proba"):
                    pm = m.predict_proba(X)
                else:
                    # Fallback: logits -> sigmoid (si decision_function)
                    if hasattr(m, "decision_function"):
                        logits = m.decision_function(X)
                        # Binario -> 2 columnas
                        p1 = 1.0 / (1.0 + np.exp(-logits))
                        pm = np.vstack([1 - p1, p1]).T
                    else:
                        # última ratio: usar predicción dura como probas 0/1
                        yhat = m.predict(X)
                        pm = np.vstack([1 - yhat, yhat]).T.astype(float)
                probs_models.append(pm)
            probs_models = _mean_probs(probs_models)  # [n_samples, 2]
            probs_ensembles.append(probs_models)
        probs_final = _mean_probs(probs_ensembles)
        return probs_final

# =========================================================
# Serialización de resultados de entrenamiento
# =========================================================
def _save_train_results(
    path_json: str,
    *,
    tipo: str,
    file: Optional[str],
    model_name: str,
    metrics: dict,
    extra: dict
):
    _ensure_dir(os.path.dirname(path_json))
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": tipo,
        "file": file,
        "model": model_name,
        **metrics,
        **extra
    }
    with open(path_json, "w") as f:
        json.dump(payload, f, indent=4)

# =========================================================
# API default_train / default_retrain
# =========================================================
def default_train(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    last_processed_file: str,
    *,
    model_instance=None,           # se usa como base_estimator
    random_state: int,
    logger,
    output_dir: str,
    param_grid: dict = None,       # no usado en IPIP
    cv: int = None                 # no usado en IPIP
):
    """
    Entrena IPIP con un split holdout 80/20.
    """
    logger.info("[TRAIN] Starting IPIP training (holdout 80/20).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    base_est = model_instance if model_instance is not None else RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=random_state
    )

    model = IPIPClassifier(
        base_estimator=base_est,
        random_state=random_state,
        prop_majoritaria=0.55,
        val_size=0.2,
        p=None, b=None
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0

    # Métricas de test
    y_pred = model.predict(X_test)
    ba = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    train_results_path = os.path.join(output_dir, "train_results.json")
    _save_train_results(
        train_results_path,
        tipo="train",
        file=last_processed_file,
        model_name="IPIPClassifier",
        metrics={
            "balanced_accuracy": ba,
            "accuracy": acc,
            "f1_macro": f1m,
            "classification_report": clf_report
        },
        extra={
            "train_size": int(len(X_train)),
            "eval_size": int(len(X_test)),
            "gridsearch": None,
            "strategy": "IPIP",
            "meta": model.meta_,
            "elapsed_sec": dt
        }
    )
    logger.info(f"[TRAIN] IPIP done. BA={ba:.4f} ACC={acc:.4f} F1={f1m:.4f} | {dt:.1f}s")

    return model, X_test, y_test, {"train_size": len(X_train), "eval_size": len(X_test), "meta": model.meta_}

def _load_previous_df(control_dir: str) -> Optional[pd.DataFrame]:
    path = Path(control_dir) / "previous_data.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def default_retrain(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    last_processed_file: str,
    model_path: str,
    *,
    mode: int,
    random_state: int,
    logger,
    output_dir: str,
    param_grid: dict = None,       # no usado en IPIP
    cv: int = None,                # no usado en IPIP
    window_size: int = None,
    replay_frac_old: float = 0.4
):
    """
    Reentrena IPIP según modo:
      0) Full retraining con todo X,y
      2) Windowed retraining (últimos window_size registros)
      5) Replay mix: mezcla X,y actual con una fracción de previous_data.csv
      6) Recalibration -> por simplicidad aplicamos full retraining (puedes sustituirlo por Platt/Isotonic sobre proba).
      Otro -> Full retraining.
    """
    logger.info(f"[RETRAIN] IPIP mode={mode}")

    # Selección de datos de reentrenamiento
    if mode == 2 and window_size is not None and window_size > 0:
        Xr = X.iloc[-window_size:, :] if isinstance(X, pd.DataFrame) else X[-window_size:]
        yr = y.iloc[-window_size:] if hasattr(y, "iloc") else y[-window_size:]
        logger.info(f"[RETRAIN] Windowed: using last {len(Xr)} rows.")
    elif mode == 5:
        prev_df = _load_previous_df(os.path.join(os.path.dirname(model_path), "..", "control"))
        if prev_df is not None and all(col in prev_df.columns for col in X.columns) and y.name in prev_df.columns:
            # Mezcla aleatoria
            prev_X = prev_df[X.columns]
            prev_y = prev_df[y.name]
            n_old = int(max(1, replay_frac_old * len(X)))
            rng = np.random.RandomState(random_state)
            sel = rng.choice(np.arange(len(prev_X)), size=min(n_old, len(prev_X)), replace=False)
            Xr = pd.concat([X, prev_X.iloc[sel]], axis=0)
            yr = pd.concat([y, prev_y.iloc[sel]], axis=0)
            logger.info(f"[RETRAIN] Replay mix: current {len(X)} + old {len(sel)} = {len(Xr)}")
        else:
            logger.warning("[RETRAIN] Replay requested but previous_data.csv incompatible/missing → using full data.")
            Xr, yr = X, y
    else:
        # 0/6/otros -> full
        Xr, yr = X, y

    # Holdout para evaluación
    X_train, X_test, y_train, y_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=random_state, stratify=yr
    )

    # Cargar current base_estimator si está guardado (opcional)
    base_estimator = None
    try:
        # No es estrictamente necesario; usamos de nuevo RF si no hay modelo
        base_estimator = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
    except Exception:
        base_estimator = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)

    model = IPIPClassifier(
        base_estimator=base_estimator,
        random_state=random_state,
        prop_majoritaria=0.55,
        val_size=0.2
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0

    y_pred = model.predict(X_test)
    ba = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    train_results_path = os.path.join(output_dir, "train_results.json")
    _save_train_results(
        train_results_path,
        tipo="retrain",
        file=last_processed_file,
        model_name="IPIPClassifier",
        metrics={
            "balanced_accuracy": ba,
            "accuracy": acc,
            "f1_macro": f1m,
            "classification_report": clf_report
        },
        extra={
            "mode": mode,
            "strategy": "IPIP",
            "train_size": int(len(X_train)),
            "eval_size": int(len(X_test)),
            "gridsearch": None,
            "elapsed_sec": dt,
            "meta": model.meta_
        }
    )
    logger.info(f"[RETRAIN] IPIP done. BA={ba:.4f} ACC={acc:.4f} F1={f1m:.4f} | {dt:.1f}s")

    return model, X_test, y_test, {"train_size": len(X_train), "eval_size": len(X_test), "meta": model.meta_}
