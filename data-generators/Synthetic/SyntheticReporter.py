import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Ruta y nombre por defecto del informe
DEFAULT_OUTPUT_DIR = "salida_tiempo_real"
REPORT_FILENAME = "report.json"


class SyntheticReporter:
    """
    Synthetic Reporter (silencioso):
    - No imprime por consola.
    - No llama a AutoVisualizer.
    - Calcula métricas esenciales y guarda un JSON que luego leerá el dashboard.
    """

    def generate_report(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        drift_type: str = "none",
        position_of_drift: Optional[int] = None,
        is_block_dataset: bool = False,
        output_dir: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        # compat: aceptar argumentos extra sin romper
        **_kwargs,
    ) -> Dict[str, Any]:
        """
        Genera un informe JSON con métricas clave y lo devuelve como dict.

        Args:
            df: DataFrame con los datos generados.
            target_col: Nombre de la columna target.
            drift_type: 'none' | 'concept' | 'data' | 'both'.
            position_of_drift: Índice donde ocurre el drift (si aplica).
            is_block_dataset: Indica si el dataset está dividido en bloques.
            output_dir: Carpeta donde guardar el JSON (por defecto 'salida_tiempo_real').
            extra_info: Información adicional a incluir en el informe.

        Returns:
            dict con el informe completo.
        """
        extra_info = extra_info or {}

        # Deducción de columna de bloque si procede
        block_col = self._detect_block_column(df) if is_block_dataset else None

        # Partes del informe
        meta = self._build_meta(df, target_col, drift_type, position_of_drift, bool(block_col))
        quality = self._calc_quality(df)
        schema = self._describe_schema(df, target_col, block_col)
        target_stats = self._target_section(df, target_col, block_col)
        correlations = self._correlation_summary(df, target_col, block_col)
        drift = self._drift_section(df, target_col, block_col, drift_type, position_of_drift)

        report: Dict[str, Any] = {
            "meta": meta,
            "schema": schema,
            "quality": quality,
            "target": target_stats,
            "correlations": correlations,
            "drift": drift,
            "extra_info": extra_info,
        }

        # Guardar JSON
        out_dir = Path((output_dir or "").strip() or DEFAULT_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / REPORT_FILENAME
        try:
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Silencioso: no levantamos, pero dejamos pista en el dict
            report.setdefault("_errors", []).append(f"Could not write report.json: {e}")

        return report

    # ---------------------------------------------------------------------
    # Secciones del informe
    # ---------------------------------------------------------------------

    def _build_meta(
        self,
        df: pd.DataFrame,
        target_col: str,
        drift_type: str,
        position_of_drift: Optional[int],
        is_block_dataset: bool,
    ) -> Dict[str, Any]:
        n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
        return {
            "rows": n_rows,
            "cols": n_cols,
            "target_col": target_col,
            "is_block_dataset": is_block_dataset,
            "drift_type": drift_type,
            "position_of_drift": int(position_of_drift) if position_of_drift is not None else None,
        }

    def _describe_schema(
        self, df: pd.DataFrame, target_col: str, block_col: Optional[str]
    ) -> Dict[str, Any]:
        cols: List[Dict[str, Any]] = []
        for col in df.columns:
            if col == target_col or col == block_col:
                continue
            cols.append(
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "n_unique": int(df[col].nunique(dropna=True)),
                }
            )
        return {
            "features_count": len(cols),
            "features": cols,
            "target_dtype": str(df[target_col].dtype) if target_col in df.columns else None,
            "block_col": block_col,
        }

    def _calc_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        n_rows, n_cols = df.shape
        total_cells = max(1, n_rows * n_cols)
        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())
        completeness = max(0.0, 100.0 - (missing / total_cells * 100.0))
        return {
            "missing_values": missing,
            "duplicate_rows": duplicates,
            "data_completeness": round(completeness, 2),
        }

    def _target_section(
        self, df: pd.DataFrame, target_col: str, block_col: Optional[str]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"has_target": target_col in df.columns}
        if target_col not in df.columns:
            return out

        # Distribución global
        vc = df[target_col].value_counts(normalize=True)
        global_dist = {str(k): float(v) for k, v in vc.items()}

        # Balance aproximado respecto a uniforme
        class_balance = None
        if len(vc) >= 2:
            p = vc.to_numpy()
            u = np.ones_like(p) / len(p)
            l1 = float(np.abs(p - u).sum())
            class_balance = max(0.0, 100.0 * (1.0 - l1 / 2.0))
        else:
            class_balance = 50.0

        out.update(
            {
                "classes_count": int(df[target_col].nunique()),
                "global_distribution": global_dist,
                "class_balance_score": round(class_balance, 2),
            }
        )

        # Distribución por bloque si aplica
        if block_col and block_col in df.columns:
            per_block: Dict[str, Dict[str, float]] = {}
            for b, g in df.groupby(block_col):
                vcb = g[target_col].value_counts(normalize=True)
                per_block[str(b)] = {str(k): float(v) for k, v in vcb.items()}
            out["per_block_distribution"] = per_block

        return out

    def _correlation_summary(
        self, df: pd.DataFrame, target_col: str, block_col: Optional[str]
    ) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in [target_col, block_col]:
            if col in numeric_cols:
                numeric_cols.remove(col)

        summary: Dict[str, Any] = {
            "numeric_features_count": len(numeric_cols),
            "top_pairs": [],
        }
        if len(numeric_cols) < 2:
            return summary

        corr = df[numeric_cols].corr()
        pairs: List[Tuple[str, str, float]] = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                v = corr.iloc[i, j]
                if not np.isnan(v):
                    pairs.append((corr.columns[i], corr.columns[j], float(abs(v))))
        top3 = sorted(pairs, key=lambda x: x[2], reverse=True)[:3]
        summary["top_pairs"] = [
            {"f1": a, "f2": b, "abs_corr": round(c, 4)} for a, b, c in top3
        ]
        return summary

    def _drift_section(
        self,
        df: pd.DataFrame,
        target_col: str,
        block_col: Optional[str],
        drift_type: str,
        position_of_drift: Optional[int],
    ) -> Dict[str, Any]:
        drift_info: Dict[str, Any] = {
            "declared_drift_type": drift_type,
            "position_of_drift": int(position_of_drift) if position_of_drift is not None else None,
        }

        if not block_col or block_col not in df.columns:
            # Sin bloques: no hay análisis entre bloques
            return drift_info

        # Concept/Data drift entre bloques contiguos
        concept_transitions: List[Tuple[str, str]] = []
        data_transitions: List[Tuple[str, str]] = []

        blocks = sorted(df[block_col].unique())
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in [target_col, block_col]:
            if col in numeric_cols:
                numeric_cols.remove(col)

        for i in range(len(blocks) - 1):
            b1, b2 = blocks[i], blocks[i + 1]
            g1 = df[df[block_col] == b1]
            g2 = df[df[block_col] == b2]

            # Concept drift: cambio en distribución del target
            concept_detected = False
            if target_col in df.columns:
                d1 = g1[target_col].value_counts(normalize=True).sort_index()
                d2 = g2[target_col].value_counts(normalize=True).sort_index()
                for cls in set(d1.index) | set(d2.index):
                    p1 = float(d1.get(cls, 0.0))
                    p2 = float(d2.get(cls, 0.0))
                    if abs(p2 - p1) * 100.0 > 10.0:  # Umbral 10 puntos porcentuales
                        concept_detected = True
                        break

            # Data drift: cambio en estadísticas de features numéricas
            data_detected = False
            if numeric_cols:
                # Revisar primeras 3 por economía
                for col in numeric_cols[:3]:
                    if col in g1.columns and col in g2.columns:
                        m1, m2 = g1[col].mean(), g2[col].mean()
                        s1, s2 = g1[col].std(), g2[col].std()
                        mean_change = abs(m2 - m1) / (abs(m1) + 1e-8)
                        std_change = abs(s2 - s1) / (abs(s1) + 1e-8)
                        if mean_change > 0.2 or std_change > 0.3:
                            data_detected = True
                            break

            if concept_detected:
                concept_transitions.append((str(b1), str(b2)))
            if data_detected:
                data_transitions.append((str(b1), str(b2)))

        drift_info.update(
            {
                "between_blocks": {
                    "checked_transitions": max(0, len(blocks) - 1),
                    "concept_drift_transitions": concept_transitions,
                    "data_drift_transitions": data_transitions,
                }
            }
        )
        return drift_info

    # ---------------------------------------------------------------------
    # Utilidades
    # ---------------------------------------------------------------------

    def _detect_block_column(self, df: pd.DataFrame) -> Optional[str]:
        for name in ("block", "chunk", "Block", "Chunk"):
            if name in df.columns:
                return name
        return None

    # Compat: método legado que redirige a generate_report
    def _report_dataset(self, df: pd.DataFrame, target_col: str, **kwargs):
        return self.generate_report(df=df, target_col=target_col, **kwargs)
