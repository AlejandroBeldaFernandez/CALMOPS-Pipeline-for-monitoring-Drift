import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import inspect
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from calmops.data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator
from calmops.data_generators.Real.RealGenerator import RealGenerator
from calmops.data_generators.Clinic.Clinic import ClinicGenerator
import river.datasets.synth as rv_synth

st.set_page_config(page_title="CalmOps Data Simulator", layout="wide", page_icon="🧪")

st.title("🧪 CalmOps Data Simulator")
st.markdown("Generate synthetic datasets using various simulation engines.")

# Sidebar Configuration
with st.sidebar:
    st.header("Generator Configuration")
    generator_type = st.selectbox(
        "Select Generator Engine",
        [
            "Synthetic (River)",
            "Real (from Data)",
            "Clinic (Medical)",
            "Scenario (Time Series)",
        ],
    )


# --- UI Helper Functions ---


def render_custom_dist_ui(key_prefix="global"):
    """Renders UI for defining custom distributions for columns."""
    st.markdown("##### Custom Column Distributions")
    if f"{key_prefix}_custom_cols" not in st.session_state:
        st.session_state[f"{key_prefix}_custom_cols"] = {}

    c_new1, c_new2, c_new3 = st.columns(3)
    with c_new1:
        new_col_name = st.text_input(
            "Column Name", key=f"{key_prefix}_col_name", placeholder="e.g. Age"
        )
    with c_new2:
        dist_type = st.selectbox(
            "Distribution",
            ["norm", "uniform", "binom", "poisson", "randint"],
            key=f"{key_prefix}_dist_type",
        )

    params = {}
    with c_new3:
        if dist_type == "norm":
            loc = st.number_input("Mean (loc)", value=0.0, key=f"{key_prefix}_d_loc")
            scale = st.number_input(
                "Std Dev (scale)", value=1.0, key=f"{key_prefix}_d_scale"
            )
            params = {"loc": loc, "scale": scale}
        elif dist_type == "uniform":
            loc = st.number_input("Start (loc)", value=0.0, key=f"{key_prefix}_d_loc")
            scale = st.number_input(
                "Width (scale)", value=1.0, key=f"{key_prefix}_d_scale"
            )
            params = {"loc": loc, "scale": scale}
        elif dist_type == "binom":
            n = st.number_input(
                "Trials (n)", value=1, min_value=1, key=f"{key_prefix}_d_n"
            )
            p = st.number_input(
                "Prob (p)",
                value=0.5,
                min_value=0.0,
                max_value=1.0,
                key=f"{key_prefix}_d_p",
            )
            params = {"n": int(n), "p": p}
        elif dist_type == "randint":
            low = st.number_input("Low", value=0, key=f"{key_prefix}_d_low")
            high = st.number_input("High", value=10, key=f"{key_prefix}_d_high")
            params = {"low": int(low), "high": int(high)}

    if st.button("Add Variable", key=f"{key_prefix}_add_var"):
        if new_col_name:
            full_spec = {"distribution": dist_type, **params}
            st.session_state[f"{key_prefix}_custom_cols"][new_col_name] = full_spec
            st.success(f"Added {new_col_name}")
        else:
            st.warning("Enter a column name")

    if st.session_state[f"{key_prefix}_custom_cols"]:
        st.json(st.session_state[f"{key_prefix}_custom_cols"])
        if st.button("Clear Variables", key=f"{key_prefix}_clear"):
            st.session_state[f"{key_prefix}_custom_cols"] = {}
            st.rerun()

    return st.session_state[f"{key_prefix}_custom_cols"]


def render_dynamics_ui(key_prefix="global"):
    """Renders UI for Dynamics Configuration (Feature Evolution)."""
    st.markdown("##### Dynamics (Feature Evolution)")
    if f"{key_prefix}_dynamics" not in st.session_state:
        st.session_state[f"{key_prefix}_dynamics"] = {}  # dict of col -> config

    c1, c2, c3 = st.columns(3)
    with c1:
        col = st.text_input("Feature to Evolve", key=f"{key_prefix}_dyn_col")
    with c2:
        d_type = st.selectbox(
            "Evolution Type",
            ["linear", "cycle", "sigmoid"],
            key=f"{key_prefix}_dyn_type",
        )

    d_params = {}
    with c3:
        if d_type == "linear":
            slope = st.number_input("Slope", value=0.1, key=f"{key_prefix}_dyn_slope")
            d_params = {"type": "linear", "slope": slope}
        elif d_type == "cycle":
            period = st.number_input(
                "Period", value=100, key=f"{key_prefix}_dyn_period"
            )
            amp = st.number_input("Amplitude", value=1.0, key=f"{key_prefix}_dyn_amp")
            d_params = {"type": "cycle", "period": period, "amplitude": amp}
        elif d_type == "sigmoid":
            center = st.number_input(
                "Center (Time)", value=500.0, key=f"{key_prefix}_dyn_center"
            )
            width = st.number_input("Width", value=100.0, key=f"{key_prefix}_dyn_width")
            amp = st.number_input("Amplitude", value=1.0, key=f"{key_prefix}_dyn_amp")
            d_params = {
                "type": "sigmoid",
                "center": center,
                "width": width,
                "amplitude": amp,
            }

    if st.button("Add Evolution", key=f"{key_prefix}_add_dyn"):
        if col:
            st.session_state[f"{key_prefix}_dynamics"][col] = d_params

    current_dyn = st.session_state[f"{key_prefix}_dynamics"]
    if current_dyn:
        st.write(current_dyn)
        if st.button("Clear Dynamics", key=f"{key_prefix}_clear_dyn"):
            st.session_state[f"{key_prefix}_dynamics"] = {}
            st.rerun()

    return {"evolve_features": current_dyn} if current_dyn else None


def render_drift_injection_ui(key_prefix="global"):
    """Renders UI for Drift Injection Configuration."""
    st.markdown("##### Drift Injection (Post-Processing)")
    if f"{key_prefix}_drift_inj" not in st.session_state:
        st.session_state[f"{key_prefix}_drift_inj"] = []

    drift_methods = [
        "shift_feature",
        "modify_variance",
        "inject_missing",
        "inject_outliers_global",
        "inject_outliers_local",
        "swap_block_order",
        "inject_categorical_shift",
    ]

    c1, c2 = st.columns(2)
    with c1:
        method = st.selectbox(
            "Drift Method", drift_methods, key=f"{key_prefix}_inj_method"
        )

    params = {}
    with c2:
        # Generic params input
        if method in [
            "shift_feature",
            "modify_variance",
            "inject_missing",
            "inject_outliers_local",
        ]:
            feature = st.text_input("Target Column", key=f"{key_prefix}_inj_col")
            params["feature_name"] = feature

        if method == "shift_feature":
            magnitude = st.number_input(
                "Shift Amount", value=1.0, key=f"{key_prefix}_inj_mag"
            )
            params["magnitude"] = magnitude
        elif method == "modify_variance":
            factor = st.number_input(
                "Variance Factor", value=2.0, key=f"{key_prefix}_inj_fac"
            )
            params["factor"] = factor
        elif method == "inject_missing":
            fraction = st.number_input(
                "Missing Fraction", 0.0, 1.0, 0.1, key=f"{key_prefix}_inj_frac"
            )
            params["fraction"] = fraction
        elif method == "inject_outliers_global":
            n_outliers = st.number_input(
                "Num Outliers", value=10, key=f"{key_prefix}_inj_nout"
            )
            scale = st.number_input("Scale", value=5.0, key=f"{key_prefix}_inj_scale")
            params["n_outliers"] = int(n_outliers)
            params["scale"] = scale

        # Start/End control
        start_idx = st.number_input(
            "Start Index (Optional)", value=0, key=f"{key_prefix}_inj_start"
        )
        end_idx = st.number_input(
            "End Index (Optional, 0=None)", value=0, key=f"{key_prefix}_inj_end"
        )
        if start_idx > 0:
            params["start_idx"] = start_idx
        if end_idx > 0:
            params["end_idx"] = end_idx

    if st.button("Add Drift Injection", key=f"{key_prefix}_add_inj"):
        st.session_state[f"{key_prefix}_drift_inj"].append(
            {"method": method, "params": params}
        )

    if st.session_state[f"{key_prefix}_drift_inj"]:
        for i, d in enumerate(st.session_state[f"{key_prefix}_drift_inj"]):
            st.text(f"{i + 1}. {d['method']} - {d['params']}")
        if st.button("Clear Drift Injections", key=f"{key_prefix}_clear_inj"):
            st.session_state[f"{key_prefix}_drift_inj"] = []
            st.rerun()

    return st.session_state[f"{key_prefix}_drift_inj"]


def render_synthetic_river():
    st.header("Synthetic Data Generation (River)")
    st.markdown("Generate data streams using River's synthetic datasets.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("River Simulator Props")
        # Get list of river generators
        river_gens = [
            name for name, obj in inspect.getmembers(rv_synth) if inspect.isclass(obj)
        ]
        selected_gen = st.selectbox(
            "Select Simulator",
            sorted(river_gens),
            index=river_gens.index("Agrawal") if "Agrawal" in river_gens else 0,
        )

        # --- Instantiate Generator ---
        gen_class = getattr(rv_synth, selected_gen)
        sig = inspect.signature(gen_class.__init__)

        gen_params = {}
        st.caption(f"Configuring {selected_gen} parameters:")

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            label = name.replace("_", " ").title()

            # Heuristics for input types
            if param.annotation == int or isinstance(param.default, int):
                val = param.default if param.default != inspect.Parameter.empty else 0
                # Special cases for choices
                if name == "classification_function" and selected_gen == "Agrawal":
                    gen_params[name] = st.selectbox(
                        label,
                        options=range(10),
                        index=val if isinstance(val, int) else 0,
                    )
                elif name == "classification_function" and selected_gen == "SEA":
                    gen_params[name] = st.selectbox(
                        label,
                        options=range(4),
                        index=val if isinstance(val, int) else 0,
                    )
                else:
                    gen_params[name] = st.number_input(label, value=val, step=1)

            elif param.annotation == float or isinstance(param.default, float):
                val = param.default if param.default != inspect.Parameter.empty else 0.0
                gen_params[name] = st.number_input(
                    label, value=val, step=0.01, format="%.4f"
                )

            elif param.annotation == bool or isinstance(param.default, bool):
                val = (
                    param.default if param.default != inspect.Parameter.empty else False
                )
                gen_params[name] = st.checkbox(label, value=val)

            elif name == "seed" or name == "random_state":
                gen_params[name] = st.number_input(label, value=42, step=1)

            else:
                pass  # Ignore unknowables

        st.divider()
        n_samples = st.number_input(
            "Total Samples to Generate",
            min_value=100,
            max_value=1000000,
            value=1000,
            step=100,
        )

    with col2:
        st.subheader("Unified Generator Options")

        st.markdown("**Drift Configuration**")
        drift_type = st.selectbox(
            "Drift Type", ["none", "abrupt", "gradual", "incremental", "virtual"]
        )

        drift_gen_instance = None
        drift_point = 0
        transition_width = 0
        inconsistency = 0.0

        if drift_type != "none":
            if drift_type == "virtual":
                inconsistency = st.slider("Inconsistency (Noise)", 0.0, 1.0, 0.1)
            else:
                drift_point = st.slider(
                    "Drift Start Position", 0, n_samples, int(n_samples / 2)
                )

                if drift_type == "gradual":
                    transition_width = st.slider(
                        "Transition Width", 10, n_samples // 2, 100
                    )

                # Configure Concept B
                st.markdown("#### Concept B (Drift Target)")
                selected_gen_drift = st.selectbox(
                    "Select Drift Simulator",
                    sorted(river_gens),
                    index=river_gens.index("Agrawal") if "Agrawal" in river_gens else 0,
                    key="drift_gen_select",
                )
                st.caption(f"Configuring {selected_gen_drift} (Concept B):")
                gen_class_drift = getattr(rv_synth, selected_gen_drift)
                # For simplicity in UI, we instantiate with default params or would need duplicate UI
                # Here we just use default constructor for Concept B to keep UI clean,
                # or we could duplicate the form. Let's assume defaults for now as per previous logic.
                # Or better, let's just instantiate it cleanly:
                try:
                    drift_gen_instance = gen_class_drift()
                except:
                    st.warning("Concept B used default parameters.")

        st.markdown("**General Options**")
        balance = st.checkbox("Balance Classes", value=True)
        inject_dates = st.checkbox("Inject Timestamps", value=True)
        date_start = "2024-01-01"
        if inject_dates:
            date_start = st.text_input("Start Date", value="2024-01-01")

        # Advanced Post-Processing
        st.divider()
        dynamics_conf = render_dynamics_ui("synthetic")
        st.divider()
        drift_inj_conf = render_drift_injection_ui("synthetic")

    if st.button("🚀 Generate Synthetic Data", type="primary"):
        with st.spinner("Simulating data stream..."):
            try:
                # 1. Instantiate Concept A
                stream_gen = gen_class(**gen_params)

                # 2. Instantiate Generator
                seed = gen_params.get("seed") or gen_params.get("random_state") or 42
                # Use standard DateConfig if needed via kwargs, but unified API accepts date_start
                synth_gen = SyntheticGenerator(random_state=seed)

                # 3. Call Unified API
                # Note: We pass kwargs that match SyntheticGenerator.generate signature
                df_result = synth_gen.generate(
                    generator_instance=stream_gen,
                    n_samples=n_samples,
                    drift_type=drift_type,
                    generator_instance_drift=drift_gen_instance,
                    drift_point=drift_point,
                    transition_width=transition_width,
                    inconsistency=inconsistency,  # Virtual drift param
                    balance=balance,
                    date_start=date_start if inject_dates else None,
                    drift_injection_config=drift_inj_conf,
                    dynamics_config=dynamics_conf,
                    save_dataset=False,  # We handle download manually
                )

                if df_result is not None:
                    st.success(f"Generated {len(df_result)} samples!")
                    st.dataframe(df_result.head(100))

                    csv = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Download CSV",
                        csv,
                        f"synthetic_{selected_gen}.csv",
                        "text/csv",
                    )
                else:
                    st.error("Generation failed (returned None).")

            except Exception as e:
                st.error(f"Error during generation: {e}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("River Simulator Props")
        # Get list of river generators
        river_gens = [
            name for name, obj in inspect.getmembers(rv_synth) if inspect.isclass(obj)
        ]
        selected_gen = st.selectbox(
            "Select Simulator",
            sorted(river_gens),
            index=river_gens.index("Agrawal") if "Agrawal" in river_gens else 0,
        )

        # --- Attempt to inspect and expose common params for the selected generator ---
        gen_class = getattr(rv_synth, selected_gen)
        sig = inspect.signature(gen_class.__init__)

        gen_params = {}
        st.caption(f"Configuring {selected_gen} parameters:")

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            label = name.replace("_", " ").title()

            # Heuristics for input types
            if param.annotation == int or isinstance(param.default, int):
                val = param.default if param.default != inspect.Parameter.empty else 0
                # Special cases for choices
                if name == "classification_function" and selected_gen == "Agrawal":
                    gen_params[name] = st.selectbox(
                        label,
                        options=range(10),
                        index=val if isinstance(val, int) else 0,
                    )
                elif name == "classification_function" and selected_gen == "SEA":
                    gen_params[name] = st.selectbox(
                        label,
                        options=range(4),
                        index=val if isinstance(val, int) else 0,
                    )
                else:
                    gen_params[name] = st.number_input(label, value=val, step=1)

            elif param.annotation == float or isinstance(param.default, float):
                val = param.default if param.default != inspect.Parameter.empty else 0.0
                gen_params[name] = st.number_input(
                    label, value=val, step=0.01, format="%.4f"
                )

            elif param.annotation == bool or isinstance(param.default, bool):
                val = (
                    param.default if param.default != inspect.Parameter.empty else False
                )
                gen_params[name] = st.checkbox(label, value=val)

            elif name == "seed" or name == "random_state":
                gen_params[name] = st.number_input(label, value=42, step=1)

            else:
                # Fallback for unknown types
                # st.text_input(label, value=str(param.default))
                pass

        st.divider()
        n_samples = st.number_input(
            "Total Samples to Generate",
            min_value=100,
            max_value=1000000,
            value=1000,
            step=100,
        )

    with col2:
        st.subheader("Synthetic Generator Wrapper Options")

        st.markdown("**Drift Injection**")
        drift_type = st.selectbox(
            "Drift Type", ["none", "abrupt", "gradual", "incremental", "virtual"]
        )

        position_drift = 0
        inconsistency = 0.0
        drift_gen_params = {}
        drift_generator_instance = None

        if drift_type != "none":
            position_drift = st.slider(
                "Drift Position (Sample Index)", 0, n_samples, int(n_samples / 2)
            )
            if drift_type == "gradual":
                _ = st.slider(
                    "Gradual Width (Transition Length)", 1, n_samples // 2, 50
                )
                # We can pass width as a generic drift option if supported,
                # but SyntheticGenerator usually handles 'position_of_drift'
                # width is not directly a standard param in generate() signature displayed,
                # but 'transition_width' is supported in SyntheticGenerator!

            if drift_type == "virtual":
                # Virtual drift concept: just noise/speed change
                inconsistency = st.slider("Inconsistency (Noise)", 0.0, 1.0, 0.0)

            elif drift_type in ["abrupt", "gradual", "incremental"]:
                st.markdown("#### Concept B (Drift Target)")
                selected_gen_drift = st.selectbox(
                    "Select Drift Simulator",
                    sorted(river_gens),
                    index=river_gens.index("Agrawal") if "Agrawal" in river_gens else 0,
                    key="drift_gen_select",
                )

                # --- Inspect Concept B ---
                st.caption(f"Configuring {selected_gen_drift} (Concept B) parameters:")
                gen_class_drift = getattr(rv_synth, selected_gen_drift)
                sig_drift = inspect.signature(gen_class_drift.__init__)

                for name, param in sig_drift.parameters.items():
                    if name == "self":
                        continue
                    label = f"B: {name.replace('_', ' ').title()}"

                    if param.annotation == int or isinstance(param.default, int):
                        val = (
                            param.default
                            if param.default != inspect.Parameter.empty
                            else 0
                        )
                        # Special handling for classification functions
                        if (
                            name == "classification_function"
                            and selected_gen_drift == "Agrawal"
                        ):
                            drift_gen_params[name] = st.selectbox(
                                label,
                                range(10),
                                index=val if isinstance(val, int) else 0,
                                key=f"drift_{name}",
                            )
                        elif (
                            name == "classification_function"
                            and selected_gen_drift == "SEA"
                        ):
                            drift_gen_params[name] = st.selectbox(
                                label,
                                range(4),
                                index=val if isinstance(val, int) else 0,
                                key=f"drift_{name}",
                            )
                        else:
                            drift_gen_params[name] = st.number_input(
                                label, value=val, step=1, key=f"drift_{name}"
                            )
                    elif param.annotation == float or isinstance(param.default, float):
                        val = (
                            param.default
                            if param.default != inspect.Parameter.empty
                            else 0.0
                        )
                        drift_gen_params[name] = st.number_input(
                            label,
                            value=val,
                            step=0.01,
                            format="%.4f",
                            key=f"drift_{name}",
                        )
                    elif param.annotation == bool or isinstance(param.default, bool):
                        val = (
                            param.default
                            if param.default != inspect.Parameter.empty
                            else False
                        )
                        drift_gen_params[name] = st.checkbox(
                            label, value=val, key=f"drift_{name}"
                        )
                    elif name in ["seed", "random_state"]:
                        drift_gen_params[name] = st.number_input(
                            label, value=42, step=1, key=f"drift_{name}"
                        )

        st.markdown("**Target & Balance**")
        balance = st.checkbox("Balance Classes (Upsampling)", value=True)
        # target_col name is usually implied by generator but Wrapper allows rename?
        # wrapper default is 'target' or None. User doesn't usually need to change this for River gens

        st.markdown("**Date Injection**")
        inject_dates = st.checkbox("Inject Timestamps", value=True)
        date_start = None

        if inject_dates:
            date_start = st.text_input("Start Date (YYYY-MM-DD)", value="2024-01-01")

        # Dynamics & Post-Hoc Drift Injection Config
        st.divider()
        dynamics_conf = render_dynamics_ui("synthetic")
        st.divider()
        drift_inj_conf = render_drift_injection_ui("synthetic")

    if st.button("🚀 Generate Synthetic Data", type="primary"):
        with st.spinner("Simulating data stream..."):
            try:
                # Instantiate Concept A
                try:
                    stream_gen = gen_class(**gen_params)
                except Exception as e:
                    st.error(f"Could not instantiate {selected_gen}: {e}")
                    return

                # Instantiate Concept B if needed
                drift_generator_instance = None
                if (
                    drift_type in ["abrupt", "gradual", "incremental"]
                    and drift_gen_params
                ):
                    try:
                        drift_generator_instance = gen_class_drift(**drift_gen_params)
                    except Exception as e:
                        st.error(
                            f"Could not instantiate Concept B ({selected_gen_drift}): {e}"
                        )
                        return

                # Use CalmOps SyntheticGenerator wrapper
                # We use random state from params if available, else 42
                seed = gen_params.get("seed") or gen_params.get("random_state") or 42
                synth_gen = SyntheticGenerator(random_state=seed)

                out_file = f"synth_{selected_gen}_{int(time.time())}.csv"

                # Check if 'width' for gradual drift was set in our custom UI or default
                # width variable would be local if defined in if block.
                # Let's clean up kwargs for generate()

                # Base generate params
                gen_args = {
                    "generator_instance": stream_gen,
                    "output_path": None,
                    "filename": out_file,
                    "n_samples": n_samples,
                    "drift_type": drift_type,
                    "position_of_drift": position_drift,
                    "inconsistency": inconsistency,
                    "balance": balance,
                    "date_start": date_start,
                    "drift_injection_config": drift_inj_conf,
                    "dynamics_config": dynamics_conf,
                }

                if drift_generator_instance:
                    gen_args["drift_generator"] = drift_generator_instance

                if drift_type == "gradual":
                    # SyntheticGenerator might expect transition_width
                    # Let's map it if we had a slider for it?
                    # I didn't add the width variable to the outer scope in chunk 1,
                    # but I can assume standard or add it if the API supports it.
                    pass

                result = synth_gen.generate(**gen_args)

                if isinstance(result, pd.DataFrame):
                    st.success(f"Successfully generated {len(result)} samples!")
                    st.dataframe(result.head(100))

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rows", len(result))
                    c2.metric("Columns", len(result.columns))

                    # Try to find target
                    target_candidate = (
                        "target" if "target" in result.columns else result.columns[-1]
                    )
                    if pd.api.types.is_numeric_dtype(result[target_candidate]):
                        c3.metric(
                            f"Mean ({target_candidate})",
                            f"{result[target_candidate].mean():.2f}",
                        )

                    csv = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Download CSV",
                        csv,
                        f"{selected_gen}_synthetic.csv",
                        "text/csv",
                        key="download-csv",
                    )
                else:
                    st.error("Generator did not return a DataFrame.")

            except Exception as e:
                st.error(f"Generation failed: {e}")
                # st.exception(e)


def render_real_data():
    st.header("Real Data Augmentation")
    st.markdown(
        "Train a generative model on your uploaded data to create synthetic replicas."
    )

    uploaded_file = st.file_uploader("Upload Original CSV", type=["csv"])

    if uploaded_file:
        df_orig = pd.read_csv(uploaded_file)
        st.write("Original Data Preview:")
        st.dataframe(df_orig.head())

        c1, c2 = st.columns(2)
        with c1:
            method = st.selectbox(
                "Method",
                [
                    "tvae",
                    "ctgan",
                    "copula",
                    "cart",
                    "rf",
                    "lgbm",
                    "gmm",
                    "datasynth",
                    "resample",
                ],
            )
            n_samples = st.number_input("Samples to Generate", value=len(df_orig))

            st.markdown("**Advanced Model Parameters**")
            epochs = 300
            if method in ["tvae", "ctgan", "copula"]:
                epochs = st.number_input("Training Epochs (SDV)", value=300, step=50)

        with c2:
            target_col = st.selectbox(
                "Target Column (Optional)", [None] + list(df_orig.columns)
            )
            balance = st.checkbox("Balance Target")
            auto_report = st.checkbox("Generate Quality Report", value=True)

            st.divider()
            custom_dists = render_custom_dist_ui("real")
            st.divider()
            dynamics_conf = render_dynamics_ui("real")
            st.divider()
            drift_inj_conf = render_drift_injection_ui("real")

        if st.button("🚀 Train & Synthesize", type="primary"):
            with st.spinner(f"Training {method.upper()} and Synthesizing..."):
                try:
                    # Model config
                    model_params = {}
                    if method in ["tvae", "ctgan", "copula"]:
                        model_params["sdv_epochs"] = epochs

                    # Initialize Generator
                    gen = RealGenerator(
                        data=df_orig,
                        method=method,
                        target_col=target_col,
                        balance_target=balance,
                        auto_report=auto_report,
                        model_params=model_params,
                    )

                    # Unified Call
                    df_synth = gen.generate(
                        n_samples=n_samples,
                        output_dir=".",
                        save_dataset=False,
                        custom_distributions=custom_dists,
                        drift_injection_config=drift_inj_conf,
                        dynamics_config=dynamics_conf,
                    )

                    if df_synth is not None:
                        st.success("Synthesis complete!")
                        st.dataframe(df_synth.head(100))

                        csv = df_synth.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇ Download Synthetic Data",
                            csv,
                            f"real_synth_{method}.csv",
                            "text/csv",
                        )
                        if auto_report:
                            st.info(
                                "Quality report generated (check output directory)."
                            )

                except Exception as e:
                    st.error(f"Synthesis failed: {e}")

    uploaded_file = st.file_uploader("Upload Original CSV", type=["csv"])

    if uploaded_file:
        df_orig = pd.read_csv(uploaded_file)
        st.write("Original Data Preview:")
        st.dataframe(df_orig.head())

        c1, c2 = st.columns(2)
        with c1:
            method = st.selectbox(
                "Method",
                [
                    "tvae",
                    "ctgan",
                    "copula",
                    "cart",
                    "rf",
                    "lgbm",
                    "gmm",
                    "datasynth",
                    "resample",
                ],
            )
            n_samples = st.number_input("Samples to Generate", value=len(df_orig))

            # Advanced Model Params
            st.markdown("**Advanced Model Parameters**")
            epochs = 300
            if method in ["tvae", "ctgan", "copula"]:
                epochs = st.number_input("Training Epochs (SDV)", value=300, step=50)

        with c2:
            target_col = st.selectbox(
                "Target Column (Optional)", [None] + list(df_orig.columns)
            )
            balance = st.checkbox("Balance Target")
            auto_report = st.checkbox("Generate Quality Report", value=True)

            # --- Advanced Configs ---
            st.divider()
            custom_dists = render_custom_dist_ui("real")

            st.divider()
            dynamics_conf = render_dynamics_ui("real")

            st.divider()
            drift_inj_conf = render_drift_injection_ui("real")

        if st.button("🚀 Train & Synthesize"):
            with st.spinner(
                f"Training {method.upper()} model... this may take a while"
            ):
                try:
                    # Construct model_params
                    model_params = {}
                    if method in ["tvae", "ctgan", "copula"]:
                        model_params["epochs"] = (
                            epochs  # SDV often uses 'epochs' or 'sdv_epochs' depending on wrapper version
                        )
                        # Check RealGenerator implementation if it maps sdv_epochs -> epochs?
                        # Viewing file previously: it passes model_params directly or extracts specific keys.
                        # line 193 in generate_synthetic_data used "sdv_epochs": 500
                        model_params["sdv_epochs"] = epochs

                    gen = RealGenerator(
                        data=df_orig,
                        method=method,
                        target_col=target_col,
                        balance_target=balance,
                        auto_report=auto_report,
                        model_params=model_params,
                    )

                    # We output to a temp dir current dir to allow finding the report?
                    # RealGenerator saves strictly to output_dir

                    df_synth = gen.generate(
                        n_samples=n_samples,
                        output_dir=".",
                        save_dataset=False,
                        custom_distributions=custom_dists,
                        drift_injection_config=drift_inj_conf,
                        dynamics_config=dynamics_conf,
                    )

                    if df_synth is not None:
                        st.success("Synthesis complete!")
                        st.dataframe(df_synth.head(100))

                        csv = df_synth.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇ Download Synthetic Data",
                            csv,
                            f"synthetic_{method}.csv",
                            "text/csv",
                        )

                        if auto_report:
                            st.info(
                                "Report generation triggered (saved to disk, visualization not yet embedded here)"
                            )

                except Exception as e:
                    st.error(f"Synthesis failed: {e}")


def render_clinic():
    st.header("Clinical Data Generator")
    st.markdown(
        "Simulate complex multi-omics clinical data (Demographics, Genes, Proteins)."
    )

    c1, c2 = st.columns(2)
    with c1:
        n_samples = st.number_input("Number of Patients", min_value=10, value=100)
        control_ratio = st.slider("Control Ratio (vs Disease)", 0.0, 1.0, 0.5)
    with c2:
        n_genes = st.number_input("Number of Genes", value=100)
        n_proteins = st.number_input("Number of Proteins", value=50)
        seed_clinic = st.number_input("Random Seed", value=42)

    st.subheader("Configuration")
    gene_type = st.selectbox("Gene Expression Type", ["RNA-Seq", "Microarray"])
    custom_cols = render_custom_dist_ui("clinic")

    st.subheader("Disease Effects (Advanced)")
    st.info("Define how the disease affects specific Genes or Proteins.")

    if "disease_effects" not in st.session_state:
        st.session_state.disease_effects = []

    with st.expander("Add New Disease Effect"):
        c_eff1, c_eff2 = st.columns(2)
        with c_eff1:
            target_scope = st.selectbox("Target Data", ["Genes", "Proteins"])
            eff_name = st.text_input(
                "Effect Name",
                value=f"Effect_{len(st.session_state.disease_effects) + 1}",
            )
        with c_eff2:
            eff_type = st.selectbox(
                "Effect Type",
                ["additive_shift", "fold_change"]
                if target_scope == "Genes"
                else ["additive_shift"],
            )
            eff_val = st.number_input("Effect Value (Magnitude)", value=1.0)

        indices_str = st.text_input("Indices (comma separated)", "0, 1, 5")

        if st.button("Add Effect"):
            try:
                indices = [
                    int(x.strip())
                    for x in indices_str.split(",")
                    if x.strip().isdigit()
                ]
                if indices:
                    st.session_state.disease_effects.append(
                        {
                            "name": eff_name,
                            "target": target_scope,
                            "effect_type": eff_type,
                            "effect_value": eff_val,
                            "indices": indices,
                        }
                    )
                    st.success("Added effect")
            except:
                st.error("Invalid indices")

    if st.session_state.disease_effects:
        st.write(st.session_state.disease_effects)
        if st.button("Clear Effects"):
            st.session_state.disease_effects = []
            st.rerun()

    st.divider()
    st.markdown("### Target Variable")
    target_mode = st.radio("Target Strategy", ["Default (Diagnosis)", "Custom Weights"])
    target_weights = None
    if target_mode == "Custom Weights":
        w_age = st.slider("Weight: Age", 0.0, 1.0, 0.5)
        w_sex = st.slider("Weight: Sex", 0.0, 1.0, 0.2)
        target_weights = {"Age": w_age, "Sex": w_sex}

    if st.button("🚀 Generate Clinical Cohort", type="primary"):
        with st.spinner("Simulating clinical cohort..."):
            try:
                gen = ClinicGenerator(seed=seed_clinic)

                # Split effects
                gene_effects = [
                    e
                    for e in st.session_state.disease_effects
                    if e["target"] == "Genes"
                ]
                prot_effects = [
                    e
                    for e in st.session_state.disease_effects
                    if e["target"] == "Proteins"
                ]

                # Prepare Target Config
                target_config = None
                if target_weights:
                    target_config = {"name": "Diagnosis", "weights": target_weights}

                # Unified Call - Flattened Args
                results = gen.generate(
                    n_samples=n_samples,
                    control_disease_ratio=control_ratio,
                    # Flattened Demographics
                    custom_demographic_columns=custom_cols if custom_cols else None,
                    # Flattened Genes
                    n_genes=n_genes,
                    gene_type=gene_type,
                    gene_disease_effects=gene_effects,
                    # Flattened Proteins
                    n_proteins=n_proteins,
                    protein_disease_effects=prot_effects,
                    # Target
                    target_variable_config=target_config,
                    save_dataset=False,
                )

                if isinstance(results, dict):
                    # Merge for display
                    df_full = results["demographics"]
                    if "genes" in results:
                        df_full = df_full.join(results["genes"])
                    if "proteins" in results:
                        df_full = df_full.join(results["proteins"])

                    st.success("Generation Complete!")
                    st.dataframe(df_full.head(50))

                    csv = df_full.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Download Clinical Data",
                        csv,
                        "clinical_cohort.csv",
                        "text/csv",
                    )
                else:
                    st.error("Generator returned invalid format.")

            except Exception as e:
                st.error(f"Error: {e}")

    c1, c2 = st.columns(2)
    with c1:
        n_patients = st.number_input("Number of Patients", min_value=10, value=100)
        control_ratio = st.slider("Control Ratio (vs Disease)", 0.0, 1.0, 0.5)
    with c2:
        n_genes = st.number_input("Number of Genes", value=100)
        n_proteins = st.number_input("Number of Proteins", value=50)
        seed_clinic = st.number_input("Random Seed", value=42)

    st.subheader("Distribution & Configuration")

    # Gene Configuration
    gene_type = st.selectbox("Gene Expression Type", ["RNA-Seq", "Microarray"])

    # Custom Demographics UI
    custom_cols = render_custom_dist_ui("clinic")

    # Advanced: Disease Effects
    st.subheader("Disease Effects (Advanced)")
    st.info("Define how the disease affects specific Genes or Proteins.")

    if "disease_effects" not in st.session_state:
        st.session_state.disease_effects = []

    with st.expander("Add New Disease Effect"):
        c_eff1, c_eff2 = st.columns(2)
        with c_eff1:
            target_scope = st.selectbox("Target Data", ["Genes", "Proteins"])
            eff_name = st.text_input(
                "Effect Name",
                value=f"Effect_{len(st.session_state.disease_effects) + 1}",
            )
        with c_eff2:
            eff_type = st.selectbox(
                "Effect Type",
                ["additive_shift", "fold_change"]
                if target_scope == "Genes"
                else ["additive_shift"],
            )
            eff_val = st.number_input("Effect Value (Magnitude)", value=1.0)

        indices_str = st.text_input("Indices (comma separated)", "0, 1, 5")

        if st.button("Add Effect"):
            try:
                indices = [
                    int(x.strip())
                    for x in indices_str.split(",")
                    if x.strip().isdigit()
                ]
                if indices:
                    st.session_state.disease_effects.append(
                        {
                            "name": eff_name,
                            "target": target_scope,
                            "effect_type": eff_type,
                            "effect_value": eff_val,
                            "indices": indices,
                        }
                    )
                    st.success("Added effect")
            except:
                st.error("Invalid indices")

    if st.session_state.disease_effects:
        st.write(st.session_state.disease_effects)
        if st.button("Clear Effects"):
            st.session_state.disease_effects = []
            st.rerun()

    # Target Variable
    st.divider()
    st.markdown("### Target Variable")
    target_mode = st.radio("Target Strategy", ["Default (Diagnosis)", "Custom Weights"])
    target_weights = None
    if target_mode == "Custom Weights":
        st.info("Assign weights to demographic columns for target generation")
        w_age = st.slider("Weight: Age", 0.0, 1.0, 0.5)
        w_sex = st.slider("Weight: Sex", 0.0, 1.0, 0.2)
        target_weights = {"Age": w_age, "Sex": w_sex}

    # Generate
    if st.button("🚀 Generate Clinical Cohort", type="primary"):
        with st.spinner("Simulating clinical cohort..."):
            try:
                gen = ClinicGenerator(seed=seed_clinic)

                # Separate effects
                gene_effects = [
                    e
                    for e in st.session_state.disease_effects
                    if e["target"] == "Genes"
                ]
                prot_effects = [
                    e
                    for e in st.session_state.disease_effects
                    if e["target"] == "Proteins"
                ]

                # Configs
                gene_config = {
                    "n_genes": n_genes,
                    "gene_type": gene_type,
                    "disease_effects": gene_effects,
                }

                protein_config = {
                    "n_proteins": n_proteins,
                    "disease_effects": prot_effects,
                }

                target_config = (
                    {"name": "Diagnosis", "weights": target_weights}
                    if target_weights
                    else None
                )

                # Unified Call
                results = gen.generate(
                    n_samples=n_patients,
                    control_disease_ratio=control_ratio,
                    clinical_config={"custom_demographic_columns": custom_cols}
                    if custom_cols
                    else None,
                    gene_config=gene_config,
                    protein_config=protein_config,
                    target_variable_config=target_config,
                )

                # Results is {demographics: df, genes: df, proteins: df}
                if isinstance(results, dict):
                    # Merge for display
                    df_full = (
                        results["demographics"]
                        .join(results["genes"])
                        .join(results["proteins"])
                    )

                    st.success("Generation Complete!")
                    st.subheader("Combined Data Preview")
                    st.dataframe(df_full.head())

                    c1, c2, c3 = st.columns(3)
                    c1.download_button(
                        "⬇ Demographics",
                        results["demographics"].to_csv(index=False).encode(),
                        "demographics.csv",
                        "text/csv",
                    )
                    c2.download_button(
                        "⬇ Genes",
                        results["genes"].to_csv(index=False).encode(),
                        "genes.csv",
                        "text/csv",
                    )
                    c3.download_button(
                        "⬇ Proteins",
                        results["proteins"].to_csv(index=False).encode(),
                        "proteins.csv",
                        "text/csv",
                    )

                else:
                    st.error("Unexpected return type from generate()")

            except Exception as e:
                st.error(f"Clinical simulation failed: {e}")


def render_scenario_generator():
    st.header("Scenario Generator (Time Series & Benchmarks)")
    st.markdown(
        "Run pre-defined complex scenarios to validate pipelines or demonstrate capabilities."
    )

    scenario_type = st.selectbox(
        "Select Scenario",
        [
            "Synthetic Agrawal (Abrupt Drift)",
            "Clinic Patients (Multi-Omics with Correlations)",
            "Real Benchmark (Drift Robustness)",
            "Privacy Demonstration",
        ],
    )

    if st.button(f"🚀 Run {scenario_type}"):
        with st.spinner(f"Running {scenario_type}... check terminal for details"):
            try:
                if scenario_type == "Synthetic Agrawal (Abrupt Drift)":
                    from river.datasets import synth
                    from calmops.data_generators.Synthetic.SyntheticGenerator import (
                        SyntheticGenerator,
                    )

                    st.info("Generating Concept A (Func 0) and Concept B (Func 1)")
                    gen = SyntheticGenerator(random_state=42)

                    # Concept A
                    df0 = gen.generate(
                        generator_instance=synth.Agrawal(
                            classification_function=0, seed=42
                        ),
                        n_samples=1000,
                        filename="agrawal_0.csv",
                        output_path=".",  # Temp
                        generate_report=False,
                    )

                    # Concept B
                    df1 = gen.generate(
                        generator_instance=synth.Agrawal(
                            classification_function=1, seed=42
                        ),
                        n_samples=500,
                        filename="agrawal_1.csv",
                        output_path=".",  # Temp
                        generate_report=False,
                    )

                    st.success("Generated Datasets!")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("Concept A (Func 0)")
                        st.dataframe(df0.head())
                    with c2:
                        st.write("Concept B (Func 1)")
                        st.dataframe(df1.head())

                    # Simple Drift Plot explanation
                    st.markdown("### Drift Visualization")
                    st.line_chart(
                        pd.concat([df0["salary"], df1["salary"]], ignore_index=True)
                    )
                    st.caption("Salary distribution change (Concatenated)")

                elif scenario_type == "Clinic Patients (Multi-Omics with Correlations)":
                    from calmops.data_generators.Clinic.Clinic import ClinicGenerator

                    gen = ClinicGenerator(seed=42)

                    # Use unified generate API
                    st.info("Generating Clinical Data with Correlations...")

                    # Define simple scenario config (similar to tutorial)
                    gene_config = {
                        "n_genes": 20,
                        "gene_type": "Microarray",
                        "correlations": "block_diagonal",  # Simplified for demo
                    }

                    results = gen.generate(
                        n_samples=200,
                        control_disease_ratio=0.4,
                        gene_config=gene_config,
                        protein_config={"n_proteins": 10},
                        target_variable_config={
                            "name": "Diagnosis",
                            "weights": {"Age": 0.5},
                        },
                    )

                    # Results is a dict {demographics: df, genes: df...}
                    if isinstance(results, dict):
                        for name, df in results.items():
                            st.subheader(f"{name.capitalize()} Data")
                            st.dataframe(df.head())
                    else:
                        st.dataframe(results.head())

                elif scenario_type == "Real Benchmark (Drift Robustness)":
                    st.warning(
                        "This scenario requires the Iris dataset and trains generic models. (Simplified for Demo)"
                    )
                    from sklearn.datasets import load_iris

                    iris = load_iris(as_frame=True)
                    df = iris.frame
                    st.write("Original Data (Iris):", df.head())

                    from calmops.data_generators.Real.RealGenerator import RealGenerator

                    gen = RealGenerator(
                        data=df,
                        method="tvae",
                        auto_report=False,
                        model_params={"epochs": 50},
                    )

                    st.info("Training TVAE (fast)...")
                    synth_df = gen.generate(n_samples=200, save_dataset=False)

                    st.success("Generated Synthetic Replicas")
                    st.write(synth_df.head())

                    # Compare
                    st.subheader("Distribution Comparison (Sepal Length)")
                    chart_data = pd.DataFrame(
                        {
                            "Original": df["sepal length (cm)"],
                            "Synthetic": synth_df["sepal length (cm)"],
                        }
                    )
                    st.line_chart(chart_data)

                elif scenario_type == "Privacy Demonstration":
                    st.info("Demonstrating pseudonymization and generalization")

                    data = {
                        "Name": [f"Person_{i}" for i in range(10)],
                        "Age": np.random.randint(20, 80, 10),
                        "Salary": np.random.randint(30000, 90000, 10),
                        "City": ["Madrid", "Barcelona", "Valencia"] * 3 + ["Madrid"],
                    }
                    df = pd.DataFrame(data)
                    st.write("Original Protected Data", df)

                    from calmops.privacy.privacy import (
                        pseudonymize_columns,
                        generalize_numeric_to_ranges,
                    )

                    df_priv = pseudonymize_columns(df, ["Name"], salt="demo")
                    df_priv = generalize_numeric_to_ranges(df_priv, ["Age"], num_bins=3)

                    st.write("Privatized Data", df_priv)

            except Exception as e:
                st.error(f"Scenario failed: {e}")


# Main Routing
if generator_type == "Synthetic (River)":
    render_synthetic_river()
elif generator_type == "Real (from Data)":
    render_real_data()
elif generator_type == "Clinic (Medical)":
    render_clinic()
elif generator_type == "Scenario (Time Series)":
    render_scenario_generator()
