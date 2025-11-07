# CALMOPS – Pipeline for Monitoring Drift

**CalmOps** es un pipeline completo basado en Python para monitorizar *data drift* y *model drift* en sistemas de machine learning. Está diseñado para ser una herramienta MLOps robusta y lista para producción.

El núcleo del proyecto es un pipeline que se puede activar con nuevos datos, que luego realiza un análisis de drift y una evaluación del modelo. Los resultados se visualizan en un dashboard web basado en Streamlit.

## Características

- **Detección de Drift:** Detectores univariantes y multivariantes de data drift y model drift usando la librería `frouros`.
- **Comparativa de Modelos:** Estrategia de promoción de modelos Champion/Challenger.
- **Re-entrenamiento:** Varios modos de re-entrenamiento:
    - `full`: Re-entrenamiento completo con los nuevos datos.
    - `incremental`: Entrenamiento incremental.
    - `window`: Entrenamiento con una ventana deslizante de datos.
    - `stacking`: Stacking de modelos.
    - `replay mix`: Combinación de datos antiguos y nuevos.
    - `recalibration`: Recalibración del modelo.
- **Dashboard Interactivo:** Visualización de resultados en un dashboard de Streamlit.
- **Monitorización de Ficheros:** Monitorización del sistema de ficheros para nuevos datos usando `watchdog`.
- **Despliegue en Producción:** Soporte para despliegue en producción con PM2 y Docker.
- **Manejo de Errores:** Patrón de Circuit Breaker para evitar fallos repetidos.
- **Logging:** Logging completo para cada pipeline.

## Arquitectura

El proyecto está estructurado en varios componentes clave:

- **`monitor`:** Responsable de monitorizar el sistema de ficheros para nuevos datos y activar el pipeline principal.
- **`pipeline`:** Orquesta todo el proceso de carga de datos, preprocesamiento, detección de drift, entrenamiento/re-entrenamiento y evaluación.
- **`Detector`:** Contiene la lógica para la detección de drift.
- **`web_interface`:** Contiene el dashboard de Streamlit.
- **`data_generators`:** Scripts para generar datasets sintéticos para pruebas y desarrollo.
- **`config`:** Ficheros de configuración para el pipeline.

## Tecnologías Principales

- **Python:** Lenguaje principal del proyecto.
- **scikit-learn:** Para entrenamiento y evaluación de modelos de machine learning.
- **frouros:** Para la detección de drift.
- **Streamlit:** Para el dashboard de monitorización.
- **pandas:** Para manipulación y análisis de datos.
- **NumPy:** Para operaciones numéricas.
- **watchdog:** Para monitorizar el sistema de ficheros.
- **PM2 & Docker:** Para despliegue en producción.

## Instalación

1.  Instalar las dependencias del proyecto usando pip:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

1.  **Ejecutar el Pipeline:**

    El punto de entrada principal para ejecutar el pipeline es el script `monitor/monitor.py`.

    ```bash
    python monitor/monitor.py
    ```

    Esto iniciará el monitor del sistema de ficheros y el dashboard de Streamlit. Cuando un nuevo fichero de datos se añade al directorio de datos especificado, el pipeline se activará automáticamente.

2.  **Ver el Dashboard:**

    El dashboard de Streamlit estará disponible en `http://localhost:8501` por defecto. El puerto se puede configurar en el script `monitor/monitor.py`.

## Despliegue en Producción

El proyecto soporta despliegue en producción usando PM2 y Docker. Puedes configurar el modo de persistencia en el script `monitor/monitor.py`:

-   **PM2:** `persistence="pm2"`
-   **Docker:** `persistence="docker"`

## Convenciones de Desarrollo

- **Estructura Modular:** El proyecto está organizado en módulos con responsabilidades específicas.
- **Configuración:** El pipeline se configura a través de scripts de Python y ficheros de configuración.
- **Personalización:** El pipeline está diseñado para ser personalizable. Puedes proporcionar tus propias funciones personalizadas para entrenamiento, re-entrenamiento y estrategias de fallback.
- **Logging:** El proyecto usa el módulo `logging` para un logging completo. Cada pipeline tiene su propio fichero de log.
- **Manejo de Errores:** El proyecto incluye un manejo de errores robusto, incluyendo un patrón de Circuit Breaker para prevenir fallos repetidos.
