import os
import json
from pathlib import Path
import pandas as pd
from scipy.io import arff
import pandas as pd
from scipy.io import arff
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel("ERROR")


def read_records(control_file: Path):
    records = {}
    if control_file.exists():
        try:
            with open(control_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        records[parts[0]] = float(parts[1])
        except Exception as e:
            print(f"[ERROR] Could not read records: {e}")
    return records


def update_record(control_file: Path, file: str, mtime: float):
    with open(control_file, "a") as f:
        f.write(f"{file},{mtime}\n")
    print(f"[CONTROL] Registered {file} in {control_file}")


from pathlib import Path
from scipy.io import arff


def _load_any_dataset(path_str: str):
    """
    Load dataset by extension: .csv, .txt, .arff, .json, .xls/.xlsx, .parquet
    - For .txt, autodetect delimiter.
    - For .arff, decodes bytes to utf-8 when needed.
    """
    p = Path(path_str)
    ext = p.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(p)

    if ext == ".txt":
        # autodetect separator
        return pd.read_csv(p, sep=None, engine="python")

    if ext == ".arff":
        data, meta = arff.loadarff(p)
        df = pd.DataFrame(data)
        # decode byte strings to str if present
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].apply(lambda v: isinstance(v, (bytes, bytearray))).any():
                df[col] = df[col].apply(
                    lambda v: v.decode("utf-8")
                    if isinstance(v, (bytes, bytearray))
                    else v
                )
        return df

    if ext == ".json":
        return pd.read_json(p)

    if ext in (".xls", ".xlsx"):
        return pd.read_excel(p)

    if ext == ".parquet":
        return pd.read_parquet(p)

    raise ValueError(f"Unsupported format: {ext} for {p}")


def read_metrics(metrics_path: Path):
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not read metrics: {e}")
    return {}


def load_json(path):
    """Load a JSON file if it exists, otherwise return an empty dict."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def load_csv(path):
    """Load a CSV file if it exists, otherwise return an empty DataFrame."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_log(path):
    """Load log file content as a string. Return default text if not found."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return "No logs found."


def dashboard_data_loader(data_dir, control_dir):
    """
    Loads the last processed dataset based on the control file.
    Supports multiple file formats: CSV, TXT, ARFF, JSON, Excel, Parquet.
    """
    control_file = Path(control_dir) / "control_file.txt"
    if not os.path.exists(control_file):
        return pd.DataFrame(), None

    # Read last processed file from control file
    with open(control_file, "r") as f:
        lines = f.readlines()
        if not lines:
            return pd.DataFrame(), None
        last_file = lines[-1].split(",")[0]

    dataset_path = os.path.join(data_dir, last_file)
    if not os.path.exists(dataset_path):
        return pd.DataFrame(), None

    try:
        # Handle each supported file type
        if dataset_path.endswith(".csv"):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith(".txt"):
            df = pd.read_csv(dataset_path, sep=None, engine="python")
        elif dataset_path.endswith(".arff"):
            data, meta = arff.loadarff(dataset_path)
            df = pd.DataFrame(data)
            # Decode byte strings to regular strings if needed
            for col in df.select_dtypes([object]):
                if df[col].apply(lambda x: isinstance(x, bytes)).any():
                    df[col] = df[col].apply(
                        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                    )
        elif dataset_path.endswith(".json"):
            df = pd.read_json(dataset_path)
        elif dataset_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(dataset_path)
        elif dataset_path.endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            st.error(f"Unsupported format: {dataset_path}")
            return pd.DataFrame(), last_file
    except Exception as e:
        st.error(f"Error loading {last_file}: {e}")
        return pd.DataFrame(), last_file

    return df, last_file


def show_evolution_section(pipeline_name: str):
    """
    Displays the 'Evolution' tab using `metrics/history.json`.
    Visualizes evaluation metrics and drift status over time.
    """
    st.subheader("🧬 Evolution History")

    # Locate history.json
    # Assuming project_root relative to this file?
    # utils.py is in calmops/web_interface/utils.py
    # project_root is calmops/
    # Actually get_pipelines_root() is not imported here?
    # dashboard files import it from calmops.utils.
    # We can pass metrics_dir or resolve it here if we import get_pipelines_root.

    # Better to pass metrics_dir as arg?
    # But dashboard functions usually take pipeline_name.
    # Let's import get_pipelines_root inside function to avoid circular imports?
    # calmops.web_interface.utils vs calmops.utils
    from calmops.utils import get_pipelines_root

    project_root = get_pipelines_root()
    metrics_dir = project_root / "pipelines" / pipeline_name / "metrics"
    history_path = metrics_dir / "history.json"

    if not history_path.exists():
        st.info(
            "No `history.json` found. Evolution history will appear here after pipeline runs."
        )
        return

    try:
        with open(history_path, "r") as f:
            history_data = json.load(f)
    except Exception as e:
        st.error(f"Could not load history.json: {e}")
        return

    if not history_data:
        st.info("History is empty.")
        return

    # Convert to DataFrame
    df_hist = pd.DataFrame(history_data)

    if df_hist.empty:
        st.info("History DataFrame is empty.")
        return

    # Convert timestamp to datetime
    if "timestamp" in df_hist.columns:
        df_hist["datetime"] = pd.to_datetime(df_hist["timestamp"], unit="s")
        df_hist = df_hist.sort_values("datetime")

    # Display raw table in expander
    with st.expander("Raw History Data"):
        st.dataframe(df_hist)

    # --- Plot 1: Evaluation Metrics Evolution ---
    # Metrics are nested in 'eval_metrics' -> 'metrics' dict or flat?
    # Based on pipeline code:
    # history_record["eval_metrics"] = eval_data[-1] (which is a dict with "metrics": {...})
    # So we need to flatten

    eval_rows = []
    for i, row in df_hist.iterrows():
        base_info = {
            "datetime": row.get("datetime"),
            "batch_id": row.get("batch_id"),
            "approved": row.get("approved"),
        }

        # Flatten eval metrics
        eval_m = row.get("eval_metrics", {})
        if isinstance(eval_m, dict):
            metrics_dict = eval_m.get("metrics", {})
            if isinstance(metrics_dict, dict):
                for k, v in metrics_dict.items():
                    if isinstance(v, (int, float)):
                        base_info[f"eval_{k}"] = v

        # Flatten drift metrics (if any)
        drift_m = row.get("drift_metrics", {})
        if isinstance(drift_m, dict):
            # global drift?
            base_info["drift_detected"] = drift_m.get("drift_detected")
            # Maybe detailed drift counts

        eval_rows.append(base_info)

    df_plot = pd.DataFrame(eval_rows)

    if not df_plot.empty:
        # Plot Evaluation Metrics
        eval_cols = [c for c in df_plot.columns if c.startswith("eval_")]
        if eval_cols:
            st.markdown("### 📈 Evaluation Metrics Over Time")
            selected_metrics = st.multiselect(
                "Select metrics to plot", eval_cols, default=eval_cols[:2]
            )

            if selected_metrics:
                fig = px.line(
                    df_plot,
                    x="datetime",
                    y=selected_metrics,
                    markers=True,
                    hover_data=["batch_id", "approved"],
                    title="Evaluation Metrics Evolution",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Plot Approval Status
        if "approved" in df_plot.columns:
            st.markdown("### ✅ Approval Status")
            # Convert boolean to int for plotting or use color
            df_plot["approval_status"] = df_plot["approved"].apply(
                lambda x: "Approved" if x else "Rejected"
            )
            color_map = {"Approved": "green", "Rejected": "red"}

            fig_app = px.scatter(
                df_plot,
                x="datetime",
                y="approval_status",
                color="approval_status",
                color_discrete_map=color_map,
                hover_data=["batch_id"],
                title="Model Approval History",
            )
            st.plotly_chart(fig_app, use_container_width=True)

    else:
        st.info("Could not extract plotting data from history.")
