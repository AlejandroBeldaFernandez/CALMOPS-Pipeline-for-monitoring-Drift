#!/usr/bin/env python3
"""
CALMOPS - Dashboard Simplificado y Estable
=========================================

Dashboard simplificado para visualización en tiempo real de datos sintéticos.
Versión estable sin auto-refresh agresivo.

Author: CalmOps Team
Version: 2.1 - Stable
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class GeneradorSimple:
    """Generador simplificado para datos sintéticos"""
    
    def __init__(self):
        # Inicializar session state de forma más robusta
        self.init_session_state()
        
    def init_session_state(self):
        """Inicializa variables de sesión de forma segura"""
        defaults = {
            'datos_acumulados': [],
            'contador_muestras': 0,
            'generando': False,
            'batch_size': 25,
            'baseline_features': None,
            'ultima_generacion': 0
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def generar_datos(self, n_samples=25):
        """Genera datos sintéticos simples"""
        try:
            # Generar datos con make_classification
            X, y = make_classification(
                n_samples=n_samples,
                n_features=4,
                n_redundant=0,
                n_informative=4,
                n_clusters_per_class=1,
                random_state=st.session_state.contador_muestras
            )
            
            # Crear registros
            timestamp = datetime.now()
            registros = []
            for i in range(len(X)):
                registro = {
                    'timestamp': timestamp + timedelta(milliseconds=i*50),
                    'sample_id': st.session_state.contador_muestras + i,
                    'feature_0': X[i, 0],
                    'feature_1': X[i, 1],
                    'feature_2': X[i, 2],
                    'feature_3': X[i, 3],
                    'target': y[i]
                }
                registros.append(registro)
            
            # Agregar al buffer
            st.session_state.datos_acumulados.extend(registros)
            
            # Mantener buffer limitado
            max_size = 500
            if len(st.session_state.datos_acumulados) > max_size:
                st.session_state.datos_acumulados = st.session_state.datos_acumulados[-max_size:]
            
            st.session_state.contador_muestras += len(X)
            
            return pd.DataFrame(registros)
            
        except Exception as e:
            st.error(f"Error generando datos: {e}")
            return pd.DataFrame()
    
    def obtener_datos(self):
        """Obtiene todos los datos como DataFrame"""
        if not st.session_state.datos_acumulados:
            return pd.DataFrame()
        return pd.DataFrame(st.session_state.datos_acumulados)
    
    def iniciar(self):
        """Inicia la generación"""
        st.session_state.generando = True
        return True
    
    def detener(self):
        """Detiene la generación"""
        st.session_state.generando = False

class DashboardSimple:
    """Dashboard simplificado"""
    
    def __init__(self):
        self.generador = GeneradorSimple()
    
    def plot_feature_evolution(self, df):
        """Gráfico de evolución de features"""
        if df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Feature 0', 'Feature 1', 'Feature 2', 'Feature 3']
        )
        
        feature_cols = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for i, (feature, pos) in enumerate(zip(feature_cols, positions)):
            if feature in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['sample_id'],
                        y=df[feature],
                        mode='lines+markers',
                        name=feature,
                        showlegend=False
                    ),
                    row=pos[0], col=pos[1]
                )
        
        fig.update_layout(title="Evolución de Features", height=400)
        return fig
    
    def plot_target_distribution(self, df):
        """Distribución del target"""
        if df.empty or 'target' not in df.columns:
            return go.Figure()
        
        fig = px.histogram(df, x='target', title="Distribución del Target")
        return fig
    
    def plot_correlation_heatmap(self, df):
        """Mapa de correlación"""
        if df.empty:
            return go.Figure()
        
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if len(feature_cols) < 2:
            return go.Figure()
        
        corr_matrix = df[feature_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(title="Matriz de Correlación")
        return fig
    
    def plot_scatter_matrix(self, df):
        """Matriz de dispersión"""
        if df.empty:
            return go.Figure()
        
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if len(feature_cols) < 2:
            return go.Figure()
        
        # Tomar muestra para evitar sobrecarga
        df_sample = df.tail(100) if len(df) > 100 else df
        
        fig = px.scatter_matrix(
            df_sample, 
            dimensions=feature_cols[:3],  # Solo las primeras 3 para evitar sobrecarga
            color='target',
            title="Matriz de Dispersión"
        )
        return fig
    
    def plot_feature_distribution(self, df, feature_name):
        """Distribución de una feature específica"""
        if df.empty or feature_name not in df.columns:
            return go.Figure()
        
        fig = px.histogram(
            df, 
            x=feature_name, 
            color='target',
            title=f"Distribución de {feature_name}",
            marginal="box"
        )
        return fig
    
    def plot_quality_breakdown(self, quality_results):
        """Gráfico de calidad de datos"""
        if not quality_results:
            return go.Figure()
        
        metrics = list(quality_results.keys())
        values = list(quality_results.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                marker_color=['green' if v >= 80 else 'orange' if v >= 60 else 'red' for v in values]
            )
        ])
        
        fig.update_layout(
            title="Métricas de Calidad de Datos",
            yaxis_title="Puntuación",
            yaxis=dict(range=[0, 100])
        )
        return fig
    
    def plot_pca_projection(self, df):
        """Proyección PCA"""
        if df.empty:
            return go.Figure()
        
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if len(feature_cols) < 2:
            return go.Figure()
        
        # Tomar muestra para PCA
        df_sample = df.tail(200) if len(df) > 200 else df
        
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_sample[feature_cols])
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                color=df_sample['target'],
                title=f"Proyección PCA (Varianza explicada: {pca.explained_variance_ratio_.sum():.2%})",
                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
            )
            return fig
        except:
            return go.Figure()
    
    def plot_statistical_summary(self, df):
        """Resumen estadístico"""
        if df.empty:
            return go.Figure()
        
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            return go.Figure()
        
        stats_data = []
        for col in feature_cols:
            stats_data.append({
                'Feature': col,
                'Mean': df[col].mean(),
                'Std': df[col].std(),
                'Min': df[col].min(),
                'Max': df[col].max()
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        fig = go.Figure(data=[
            go.Bar(name='Mean', x=stats_df['Feature'], y=stats_df['Mean']),
            go.Bar(name='Std', x=stats_df['Feature'], y=stats_df['Std'])
        ])
        
        fig.update_layout(
            title="Resumen Estadístico por Feature",
            barmode='group'
        )
        return fig
    
    def plot_boxplots_summary(self, df):
        """Box plots para todas las features"""
        if df.empty:
            return go.Figure()
        
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            return go.Figure()
        
        fig = go.Figure()
        
        for col in feature_cols:
            fig.add_trace(go.Box(
                y=df[col],
                name=col,
                boxpoints='outliers'
            ))
        
        fig.update_layout(title="Distribución de Features (Box Plots)")
        return fig
    
    def plot_violin_plots(self, df):
        """Violin plots para features"""
        if df.empty:
            return go.Figure()
        
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            return go.Figure()
        
        fig = go.Figure()
        
        for col in feature_cols:
            fig.add_trace(go.Violin(
                y=df[col],
                name=col,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(title="Densidad de Features (Violin Plots)")
        return fig
    
    def calcular_quality_metrics(self, df):
        """Calcula métricas de calidad de datos"""
        if df.empty:
            return {
                'statistical_validity': 80,
                'feature_diversity': 80,
                'data_completeness': 100,
                'class_balance': 80,
                'overall_score': 85
            }
        
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        # Statistical validity
        statistical_validity = 85
        
        # Feature diversity
        feature_diversity = min(100, len(feature_cols) * 20)
        
        # Data completeness
        data_completeness = 100
        
        # Class balance
        if 'target' in df.columns:
            class_counts = df['target'].value_counts()
            if len(class_counts) > 1:
                imbalance_ratio = class_counts.max() / class_counts.min()
                class_balance = max(20, 100 - (imbalance_ratio - 1) * 20)
            else:
                class_balance = 50
        else:
            class_balance = 80
        
        # Overall score
        overall_score = (statistical_validity + feature_diversity + data_completeness + class_balance) / 4
        
        return {
            'statistical_validity': statistical_validity,
            'feature_diversity': feature_diversity,
            'data_completeness': data_completeness,
            'class_balance': class_balance,
            'overall_score': overall_score
        }
    
    def mostrar_dashboard(self):
        """Dashboard principal"""
        st.title("🎯 CALMOPS - Dashboard Automático")
        st.markdown("**Dashboard con generación automática continua de datos sintéticos**")
        st.markdown("*Los datos se generan automáticamente cada segundo cuando está activo.*")
        
        # Controles simples
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Iniciar", type="primary"):
                self.generador.iniciar()
                self.generador.generar_datos()
                st.success("✅ Iniciado")
        
        with col2:
            if st.button("⏹️ Detener"):
                self.generador.detener()
                st.info("Detenido")
        
        with col3:
            # Estado simple
            if st.session_state.generando:
                st.success("🟢 Generando automáticamente")
            else:
                st.error("🔴 Detenido")
        
        # Configuración
        with st.expander("⚙️ Configuración"):
            batch_size = st.slider("Tamaño de lote", 5, 100, 25)
            st.session_state.batch_size = batch_size
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Muestras", st.session_state.contador_muestras)
        with col2:
            df = self.generador.obtener_datos()
            if not df.empty and 'target' in df.columns:
                balance = df['target'].mean()
                st.metric("⚖️ Balance", f"{balance:.1%}")
        with col3:
            st.metric("📦 Buffer", len(st.session_state.datos_acumulados))
        
        # Visualizaciones
        df = self.generador.obtener_datos()
        
        if not df.empty:
            st.markdown("### 📊 Visualizaciones Completas")
            
            # SECCIÓN 1: Análisis Básico
            st.markdown("#### 🔍 Análisis Básico")
            
            # FILA 1: Quality Breakdown + Correlation Heatmap
            col1, col2 = st.columns(2)
            
            with col1:
                quality_results = self.calcular_quality_metrics(df)
                fig_quality = self.plot_quality_breakdown(quality_results)
                st.plotly_chart(fig_quality, use_container_width=True)
            
            with col2:
                fig_corr = self.plot_correlation_heatmap(df)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # FILA 2: PCA Projection + Statistical Summary
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pca = self.plot_pca_projection(df)
                st.plotly_chart(fig_pca, use_container_width=True)
            
            with col2:
                fig_stats = self.plot_statistical_summary(df)
                st.plotly_chart(fig_stats, use_container_width=True)
            
            # FILA 3: Target Distribution (ancho completo)
            fig_target = self.plot_target_distribution(df)
            st.plotly_chart(fig_target, use_container_width=True)
            
            # FILA 4: Feature Distributions individuales
            st.markdown("#### 📈 Distribuciones Individuales")
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            
            for i in range(0, len(feature_cols), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(feature_cols):
                        fig_dist = self.plot_feature_distribution(df, feature_cols[i])
                        st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    if i + 1 < len(feature_cols):
                        fig_dist = self.plot_feature_distribution(df, feature_cols[i + 1])
                        st.plotly_chart(fig_dist, use_container_width=True)
            
            # SECCIÓN 2: Análisis Exploratorio
            st.markdown("#### 🔎 Análisis Exploratorio")
            
            # FILA 1: Feature Evolution (ancho completo)
            fig_evolution = self.plot_feature_evolution(df)
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            # FILA 2: Box plots y Violin plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig_box = self.plot_boxplots_summary(df)
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                fig_violin = self.plot_violin_plots(df)
                st.plotly_chart(fig_violin, use_container_width=True)
            
            # SECCIÓN 3: Análisis Avanzado
            st.markdown("#### 🔬 Análisis Avanzado")
            
            # FILA 1: Scatter Matrix (ancho completo)
            fig_scatter = self.plot_scatter_matrix(df)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Tabla de datos recientes
            st.markdown("### 📋 Datos Recientes")
            st.dataframe(df.tail(10), use_container_width=True)
        
        else:
            st.info("Sin datos. Presiona '▶️ Iniciar' para comenzar.")
        
        # Auto-refresh automático cuando está activo
        if st.session_state.generando:
            # Generar nuevo lote automáticamente
            self.generador.generar_datos(st.session_state.batch_size)
            # Pausa fija de 1 segundo
            time.sleep(1.0)
            st.rerun()
        
        # Botón manual adicional para refrescar
        if st.button("🔄 Refrescar Visualizaciones"):
            st.rerun()

def main():
    """Función principal"""
    st.set_page_config(
        page_title="CALMOPS - Dashboard Simple",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    dashboard = DashboardSimple()
    dashboard.mostrar_dashboard()

if __name__ == "__main__":
    main()