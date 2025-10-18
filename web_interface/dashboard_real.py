# dashboard_real.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import pickle
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(layout="wide")

# --- Auto-reloading Logic ---
REPORT_FILE = ".drift_report.pkl"
REFRESH_INTERVAL_SECONDS = 30

def check_for_updates():
    """Checks if the report file has been updated and reruns the app if it has."""
    if not os.path.exists(REPORT_FILE):
        return

    try:
        current_mtime = os.path.getmtime(REPORT_FILE)

        if "last_mtime" not in st.session_state:
            st.session_state.last_mtime = current_mtime

        if current_mtime > st.session_state.last_mtime:
            st.session_state.last_mtime = current_mtime
            st.toast("Report file updated. Refreshing dashboard...")
            # A short sleep helps ensure the toast message is rendered.
            time.sleep(1)
            st.rerun()
    except FileNotFoundError:
        if "last_mtime" in st.session_state:
            del st.session_state.last_mtime
    except Exception as e:
        st.warning(f"Could not check for file updates: {e}")

check_for_updates()
# --- End of Auto-reloading Logic ---

st.title("🔬 Visualizador de Reporte de Drift y Análisis de Datos")

# Add a manual refresh button as a fallback
if st.button("Recargar Datos Manualmente"):
    st.rerun()

# Define the path for the temporary report file
REPORT_FILE_PATH = ".drift_report.pkl"

def display_drift_report(report):
    """Helper function to display the drift analysis report contents."""
    st.header("Resultados del Análisis de Drift")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Drift", "Evolución Temporal", "Análisis PCA", "Label Shift", "Concept Drift"])

    with tab1:
        st.subheader("Comparación de la Distribución de Características")
        data_drift_plots = report.get("data_drift_plots", {})
        if data_drift_plots:
            for feature, fig in data_drift_plots.items():
                st.markdown(f"#### Característica: `{feature}`")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No se generaron gráficos de deriva de datos.")

    with tab2:
        st.subheader("Evolución de Características a lo largo del Tiempo")
        feature_evolution_plots = report.get("feature_evolution_plots", {})
        if feature_evolution_plots:
            for feature, fig in feature_evolution_plots.items():
                st.markdown(f"#### Característica: `{feature}`")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No se generaron gráficos de evolución temporal (requiere una columna de fecha/hora).")

    with tab3:
        st.subheader("Comparación de Estructura de Datos (PCA)")
        pca_plot = report.get("pca_comparison_plot")
        if pca_plot:
            st.pyplot(pca_plot)
            plt.close(pca_plot)
        else:
            st.info("No se generó ningún gráfico de comparación PCA.")

    with tab4:
        st.subheader("Comparación de la Distribución de la Etiqueta (Target)")
        label_shift_plot = report.get("label_shift_plot")
        if label_shift_plot:
            st.pyplot(label_shift_plot)
            plt.close(label_shift_plot)
        else:
            st.info("No se generó ningún gráfico de deriva de etiqueta.")

    with tab5:
        st.subheader("Análisis de la Relación entre Características y Etiqueta")
        concept_drift_plots = report.get("concept_drift_plots", {})
        if concept_drift_plots:
            for feature, fig in concept_drift_plots.items():
                st.markdown(f"#### Característica: `{feature}`")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No se generaron gráficos de deriva de concepto.")

def display_comprehensive_report(report):
    """Helper function to display the comprehensive analysis report contents."""
    st.header("Análisis Comparativo Completo (Antes vs. Después del Drift)")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Info Básica", "Análisis Estadístico", "Análisis de Features", "Calidad SDV", "Visualizaciones", "Evaluación de Escalado"])

    with tab1:
        st.subheader("Información Básica del Dataset")
        basic_info = report.get("basic_info", {})
        if basic_info:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Dataset Original (Antes del Drift)**")
                st.json(basic_info.get("real_dataset", {}))
            with col2:
                st.markdown("**Dataset Actual (Después del Drift)**")
                st.json(basic_info.get("synthetic_dataset", {}))
        else:
            st.info("No hay información básica disponible.")

    with tab2:
        st.subheader("Análisis Estadístico Detallado")
        stats_analysis = report.get("statistical_analysis", {})
        if stats_analysis:
            st.markdown("**Resumen de Características Numéricas**")
            if "numeric_stats" in stats_analysis and "real_summary" in stats_analysis["numeric_stats"]:
                st.markdown("***Dataset Original***")
                st.dataframe(pd.DataFrame(stats_analysis["numeric_stats"]["real_summary"]))
                st.markdown("***Dataset Actual***")
                st.dataframe(pd.DataFrame(stats_analysis["numeric_stats"]["synthetic_summary"]))
            
            st.markdown("**Pruebas Estadísticas (Numéricas)**")
            if "numeric_stats" in stats_analysis and "statistical_tests" in stats_analysis["numeric_stats"]:
                st.dataframe(pd.DataFrame(stats_analysis["numeric_stats"]["statistical_tests"]).T)

            st.markdown("**Resumen de Características Categóricas**")
            if "categorical_stats" in stats_analysis:
                st.json(stats_analysis["categorical_stats"])
        else:
            st.info("No hay análisis estadístico disponible.")

    with tab3:
        st.subheader("Análisis de Características")
        feature_analysis = report.get("feature_analysis", {})
        if feature_analysis:
            st.markdown("**Análisis de Correlación**")
            st.json(feature_analysis.get("correlation", {}))
            st.markdown("**Análisis de Rangos de Características**")
            st.dataframe(pd.DataFrame(feature_analysis.get("feature_ranges", {})).T)
        else:
            st.info("No hay análisis de características disponible.")

    with tab4:
        st.subheader("Evaluación de Calidad de Datos (SDV)")
        sdv_quality = report.get("sdv_quality", {})
        if sdv_quality:
            st.json(sdv_quality)
        else:
            st.info("No hay evaluación de calidad SDV disponible.")

    with tab5:
        st.subheader("Visualizaciones Adicionales")
        visualizations = report.get("visualizations", {})
        if visualizations:
            for name, fig in visualizations.items():
                st.markdown(f"#### {name.replace('_', ' ').title()}")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No hay visualizaciones adicionales disponibles.")

    with tab6:
        st.subheader("Evaluación de Escalado del Generador")
        scaling_plot = report.get("scaling_plot")
        if scaling_plot:
            st.pyplot(scaling_plot)
            plt.close(scaling_plot)
        else:
            st.info("No se generó ningún gráfico de evaluación de escalado. Esto puede requerir que se pase un `RealGeneratorClass` al reporter.")

# --- Main Logic ---
if os.path.exists(REPORT_FILE_PATH):
    with st.spinner("Cargando reporte pre-calculado..."):
        try:
            with open(REPORT_FILE_PATH, 'rb') as f:
                combined_report = pickle.load(f)
            
            # Display both reports
            display_drift_report(combined_report.get("drift_analysis", {}))
            st.divider()
            display_comprehensive_report(combined_report.get("comprehensive_analysis", {}))

        except Exception as e:
            st.error(f"Error al cargar el fichero del reporte: {e}")
            st.info(f"Por favor, ejecuta el script 'analysis_runner.py' de nuevo para generar el reporte.")

else:
    st.warning("No se ha encontrado ningún reporte.")
    st.info("Por favor, ejecuta el script `analysis_runner.py` para generar un análisis y poder visualizarlo aquí.")

# --- Polling mechanism for auto-refresh ---
# This part of the script will execute after the page is rendered.
# We sleep for the desired interval and then trigger a rerun to check for file updates again.
time.sleep(REFRESH_INTERVAL_SECONDS)
st.rerun()