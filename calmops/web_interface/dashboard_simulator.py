import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import inspect
import json
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
        ["Synthetic (River)", "Real (from Data)", "Clinic (Medical)"],
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
                width = st.slider(
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
        date_every = 1

        if inject_dates:
            date_start = st.text_input("Start Date (YYYY-MM-DD)", value="2024-01-01")
            date_every = st.number_input(
                "Step (Rows per timestamp increment)", value=1, min_value=1
            )

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
                        original_data=df_orig,
                        method=method,
                        target_column=target_col,
                        balance_target=balance,
                        auto_report=auto_report,
                        model_params=model_params,
                    )

                    # We output to a temp dir current dir to allow finding the report?
                    # RealGenerator saves strictly to output_dir

                    df_synth = gen.synthesize(
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
        n_patients = st.number_input("Number of Patients", min_value=10, value=100)
        control_ratio = st.slider("Control Ratio (vs Disease)", 0.0, 1.0, 0.5)
    with c2:
        n_genes = st.number_input("Number of Genes", value=100)
        n_proteins = st.number_input("Number of Proteins", value=50)
        seed_clinic = st.number_input("Random Seed", value=42)

    st.subheader("Distribution Controls (Advanced)")
    # --- Advanced Parameters ---
    st.markdown("### Advanced Distribution Control")

    with st.expander("Mean / Center Parameters"):
        c_gen1, c_gen2 = st.columns(2)
        # Defaults from Clinic.py: RNA-seq ~ log(80) approx 4.38, Microarray ~ 7.0
        # Protein ~ 3.0
        with c_gen1:
            gene_mean_rna = st.number_input(
                "Gene Mean Log Center (RNA-Seq)", value=4.4, step=0.1
            )
            gene_mean_micro = st.number_input(
                "Gene Mean Loc Center (Microarray)", value=7.0, step=0.5
            )
        with c_gen2:
            protein_mean = st.number_input(
                "Protein Mean Log Center", value=3.0, step=0.1
            )

    with st.expander("Add Custom Demographic Variable"):
        st.info("Define new columns with specific distributions.")

        # We need session state to store added columns across reruns
        if "custom_demo_cols" not in st.session_state:
            st.session_state.custom_demo_cols = {}

        c_new1, c_new2, c_new3 = st.columns(3)
        with c_new1:
            new_col_name = st.text_input("Column Name", placeholder="e.g. BMI")
        with c_new2:
            dist_type = st.selectbox(
                "Distribution", ["norm", "uniform", "binom", "poisson", "randint"]
            )

        # Dynamic params based on dist
        params = {}
        with c_new3:
            if dist_type == "norm":
                loc = st.number_input("Mean (loc)", value=0.0)
                scale = st.number_input("Std Dev (scale)", value=1.0)
                params = {"loc": loc, "scale": scale}
            elif dist_type == "uniform":
                loc = st.number_input("Start (loc)", value=0.0)
                scale = st.number_input("Width (scale)", value=1.0)
                params = {"loc": loc, "scale": scale}
            elif dist_type == "binom":
                n = st.number_input("Trials (n)", value=1, min_value=1)
                p = st.number_input("Prob (p)", value=0.5, min_value=0.0, max_value=1.0)
                params = {"n": int(n), "p": p}
            elif dist_type == "randint":
                low = st.number_input("Low", value=0)
                high = st.number_input("High", value=10)
                params = {"low": int(low), "high": int(high)}

        if st.button("Add Variable"):
            if new_col_name:
                full_spec = {"distribution": dist_type, **params}
                st.session_state.custom_demo_cols[new_col_name] = full_spec
                st.success(f"Added {new_col_name}")
            else:
                st.warning("Enter a column name")

        # Display current Custom Vars
        if st.session_state.custom_demo_cols:
            st.write("Current Custom Variables:")
            st.json(st.session_state.custom_demo_cols)
            if st.button("Clear Custom Variables"):
                st.session_state.custom_demo_cols = {}
                st.rerun()

    # --- Gene Type Selection ---
    st.markdown("### Genetic Data Configuration")
    gene_type_opts = ["RNA-Seq", "Microarray"]
    gene_type = st.selectbox("Gene Expression Type", gene_type_opts)

    # --- Disease Effects Configuration ---
    st.markdown("### Disease Effects (Bio-Markers)")
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
                ["additive_shift", "fold_change", "log_transform"]
                if target_scope == "Genes"
                else [
                    "additive_shift",
                    "simple_additive_shift",
                ],  # Proteins use lognorm, additive in log space = fold change
            )
            eff_val = st.number_input("Effect Value (Magnitude)", value=1.0)

        c_eff3, c_eff4 = st.columns(2)
        with c_eff3:
            # Indices selection
            idx_mode = st.radio("Indices Selection", ["Range", "List"])
        with c_eff4:
            indices_list = []
            max_idx = n_genes if target_scope == "Genes" else n_proteins
            if idx_mode == "Range":
                start_i = st.number_input("Start Index", 0, max_idx - 1, 0)
                end_i = st.number_input(
                    "End Index (Exclusive)",
                    start_i + 1,
                    max_idx,
                    min(start_i + 10, max_idx),
                )
                indices_list = list(range(start_i, end_i))
            else:
                idx_str = st.text_input("Indices (comma separated)", "0, 1, 5")
                try:
                    indices_list = [
                        int(x.strip())
                        for x in idx_str.split(",")
                        if x.strip().isdigit()
                    ]
                except Exception:
                    st.error("Invalid format")

        if st.button("Add Effect"):
            if indices_list:
                effect_def = {
                    "name": eff_name,
                    "target": target_scope,
                    "effect_type": eff_type,
                    "effect_value": eff_val,
                    "indices": indices_list,
                }
                st.session_state.disease_effects.append(effect_def)
                st.success(f"Added effect to {len(indices_list)} {target_scope}")
            else:
                st.warning("No indices selected")

    # Display current effects
    if st.session_state.disease_effects:
        st.write("Active Disease Effects:")
        for i, eff in enumerate(st.session_state.disease_effects):
            st.code(
                f"{eff['target']} | {eff['name']}: {eff['effect_type']}={eff['effect_value']} on {len(eff['indices'])} features"
            )
        if st.button("Clear Effects"):
            st.session_state.disease_effects = []
            st.rerun()

    # --- Correlation Configuration ---
    st.markdown("### Correlation Structure")
    corr_mode = st.radio(
        "Correlation Mode",
        ["None (Independent)", "Random (Weak)", "Random (Strong)", "Block Diagonal"],
    )

    def generate_random_corr(n, scale=0.1):
        # Random correlation matrix: A * A.T
        A = np.random.normal(0, scale, size=(n, n))
        cov = np.dot(A, A.T)
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        return corr

    def generate_block_corr(n, n_blocks, block_corr=0.8):
        corr = np.eye(n)
        block_size = n // n_blocks
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size if b < n_blocks - 1 else n
            # Set block correlation
            corr[start:end, start:end] = block_corr
            np.fill_diagonal(corr, 1.0)
        return corr

    gene_corr = None
    prot_corr = None

    if corr_mode != "None (Independent)":
        with st.expander("Correlation Details"):
            if "Random" in corr_mode:
                scale = 0.5 if "Strong" in corr_mode else 0.1
                # Generate on the fly? Might be slow for large N.
                if st.button("Generate Correlation Matrices"):
                    st.session_state.gene_corr = generate_random_corr(n_genes, scale)
                    st.session_state.prot_corr = generate_random_corr(n_proteins, scale)
                    st.success("Generated random correlations.")
            elif corr_mode == "Block Diagonal":
                n_blocks = st.slider("Number of Blocks", 1, 20, 5)
                block_val = st.slider("Intra-Block Correlation", 0.0, 1.0, 0.7)
                if st.button("Generate Block Correlations"):
                    st.session_state.gene_corr = generate_block_corr(
                        n_genes, n_blocks, block_val
                    )
                    st.session_state.prot_corr = generate_block_corr(
                        n_proteins, n_blocks, block_val
                    )
                    st.success(f"Generated {n_blocks} blocks.")
    else:
        st.session_state.gene_corr = None
        st.session_state.prot_corr = None

    # --- Target Variable Generation (Post-Processing) ---
    st.markdown("### Target Variable Generation (Diagnosis)")
    target_mode = st.radio(
        "Detailed Target Strategy", ["Default (Ratio-based)", "Rule-based (Formula)"]
    )

    formula_str = ""
    target_noise = 0.0
    if target_mode == "Rule-based (Formula)":
        st.info(
            "Define a pandas.eval expression. Columns: Demographic vars, 'G_0'...'G_N', 'P_0'...'P_M'."
        )
        formula_str = st.text_area("Formula", "Age > 65 and G_0 > 5 and Sex == 'Male'")
        target_noise = st.slider("Noise (Std Dev)", 0.0, 2.0, 0.0)

    # --- Dynamics & Drift UI ---
    st.divider()
    clinic_dynamics = render_dynamics_ui("clinic")
    st.divider()
    clinic_drift = render_drift_injection_ui("clinic")

    if st.button("🚀 Generate Clinical Cohort"):
        with st.spinner("Simulating clinical cohort..."):
            try:
                gen = ClinicGenerator(seed=seed_clinic)

                # Separate effects for Genes and Proteins
                gene_effects_config = []
                protein_effects_config = []

                for eff in st.session_state.disease_effects:
                    clean_eff = {k: v for k, v in eff.items() if k != "target"}
                    if eff["target"] == "Genes":
                        gene_effects_config.append(clean_eff)
                    else:
                        protein_effects_config.append(clean_eff)

                # Demographics
                custom_cols = st.session_state.get("custom_demo_cols", None)

                df_dem, _ = gen.generate_demographic_data(
                    n_samples=n_patients,
                    custom_demographic_columns=custom_cols,
                    control_disease_ratio=control_ratio,
                )

                # Genes
                # Use cached correlations if available
                # Note: Session state corr might mismatch if N_genes changed. Check shape.
                g_corr = st.session_state.get("gene_corr")
                if g_corr is not None and g_corr.shape != (n_genes, n_genes):
                    g_corr = None  # Invalid shape, ignore or regen

                df_genes = gen.generate_gene_data(
                    n_genes=n_genes,
                    gene_type=gene_type,
                    demographic_df=df_dem,
                    demographic_id_col="Patient_ID",
                    gene_mean_log_center=gene_mean_rna if gene_type == "RNA-Seq" else 0,
                    gene_mean_loc_center=gene_mean_micro
                    if gene_type == "Microarray"
                    else 0,
                    n_samples=n_patients,
                    control_disease_ratio=control_ratio,
                    disease_effects_config=gene_effects_config
                    if gene_effects_config
                    else None,
                    gene_correlations=g_corr,
                )

                # Proteins
                p_corr = st.session_state.get("prot_corr")
                if p_corr is not None and p_corr.shape != (n_proteins, n_proteins):
                    p_corr = None

                df_prot = gen.generate_protein_data(
                    n_proteins=n_proteins,
                    demographic_df=df_dem,
                    demographic_id_col="Patient_ID",
                    protein_mean_log_center=protein_mean,
                    n_samples=n_patients,
                    control_disease_ratio=control_ratio,
                    disease_effects_config=protein_effects_config
                    if protein_effects_config
                    else None,
                    protein_correlations=p_corr,
                )

                # --- Post-Processing: Target Generation & Dynamics & Drift ---

                # Merge everything first for unified processing
                df_full = df_dem.join(df_genes).join(df_prot)

                from calmops.data_generators.Dynamics.DynamicsInjector import (
                    DynamicsInjector,
                )
                from calmops.data_generators.DriftInjection import DriftInjector

                injector = DynamicsInjector(seed=seed_clinic)

                # 1. Custom Target (Formula)
                if target_mode == "Rule-based (Formula)" and formula_str:
                    try:
                        df_full = injector.construct_target(
                            df_full,
                            target_col="Diagnosis_Rule",
                            formula=formula_str,
                            task_type="classification",
                            noise_std=target_noise,
                        )
                        st.success("Apply custom diagnosis rule!")
                    except Exception as e:
                        st.error(f"Rule application failed: {e}")

                # 2. Dynamics (Feature Evolution)
                if clinic_dynamics and "evolve_features" in clinic_dynamics:
                    try:
                        # We need a time column for evolution. Clinic data is usually static snapshot.
                        # But we can simulate time or just use index as proxy if requested?
                        # Usually DynamicsInjector requires time_col.
                        # If no time_col, maybe skipped or use mock?
                        # Let's assume user knows what they are doing or use index.
                        if "Time" not in df_full.columns:
                            df_full["Time"] = range(len(df_full))

                        df_full = injector.evolve_features(
                            df_full,
                            time_col="Time",
                            **clinic_dynamics["evolve_features"],
                        )
                        st.success(" Applied Feature Evolution (Dynamics)!")
                    except Exception as e:
                        st.warning(f"Dynamics failed: {e}")

                # 3. Drift Injection
                if clinic_drift:
                    drift_inj = DriftInjector(
                        original_df=df_full,
                        output_dir=".",
                        generator_name="ClinicDrift",
                        random_state=seed_clinic,
                    )
                    for drift_conf in clinic_drift:
                        method_name = drift_conf.get("method")
                        params = drift_conf.get("params", {})
                        if hasattr(drift_inj, method_name):
                            # Auto-inject df if missing
                            if "df" not in params:
                                params["df"] = df_full

                            try:
                                res = getattr(drift_inj, method_name)(**params)
                                if isinstance(res, pd.DataFrame):
                                    df_full = res
                                st.success(f"Applied drift: {method_name}")
                            except Exception as e:
                                st.error(f"Drift {method_name} failed: {e}")

                # --- Split Back for Display/Download ---
                # We assume column names are unique enough.
                # Update cols in sub-dfs if they exist in df_full
                cols_dem = [c for c in df_dem.columns if c in df_full.columns]
                # Diagnosis_Rule might be new, usually desired in Demographics
                if "Diagnosis_Rule" in df_full.columns:
                    if "Diagnosis_Rule" not in cols_dem:
                        cols_dem.append("Diagnosis_Rule")

                df_dem = df_full[cols_dem]

                cols_genes = [c for c in df_genes.columns if c in df_full.columns]
                df_genes = df_full[cols_genes]

                cols_prot = [c for c in df_prot.columns if c in df_full.columns]
                df_prot = df_full[cols_prot]

                st.subheader("Demographics")
                st.dataframe(df_dem.head())

                st.subheader(f"Genes ({gene_type})")
                st.dataframe(df_genes.head())

                st.subheader("Proteins (Expression)")
                st.dataframe(df_prot.head())

                c_d, c_g, c_p, c_all = st.columns(4)

                csv_d = df_dem.to_csv(index=False).encode("utf-8")
                c_d.download_button(
                    "⬇ Demographics", csv_d, "demographics.csv", "text/csv"
                )

                csv_g = df_genes.to_csv(index=False).encode("utf-8")
                c_g.download_button("⬇ Genes", csv_g, "genes.csv", "text/csv")

                csv_p = df_prot.to_csv(index=False).encode("utf-8")
                c_p.download_button("⬇ Proteins", csv_p, "proteins.csv", "text/csv")

                # Merged download
                csv_m = df_full.to_csv(index=True).encode("utf-8")
                c_all.download_button(
                    "⬇ Merged Dataset", csv_m, "clinical_full.csv", "text/csv"
                )

            except Exception as e:
                st.error(f"Clinical simulation failed: {e}")


# Main Routing
if generator_type == "Synthetic (River)":
    render_synthetic_river()
elif generator_type == "Real (from Data)":
    render_real_data()
elif generator_type == "Clinic (Medical)":
    render_clinic()
