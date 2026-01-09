# 🧠 Conceptos Básicos de CalmOps

Saber programar no es suficiente para usar CalmOps con éxito. Necesitas entender **qué** estás monitoreando. 

Esta guía explica los conceptos de MLOps de forma sencilla, sin ecuaciones matemáticas.

---

## 1. ¿Qué es Data Drift? (La "Realidad" cambia)

Imagina que entrenas un modelo para detectar **gatos** usando solo fotos de gatos **negros**.
El modelo aprende: *"Gato = Animal pequeño y negro"*.

Un mes después, tus usuarios empiezan a subir fotos de gatos **blancos**. 
El modelo fallará, no porque se haya "roto", sino porque los datos actuales (gatos blancos) son diferentes a los datos de entrenamiento (gatos negros).

Esto es **Data Drift** (o *Covariate Shift*).
*   **En resumen:** La entrada ($X$) cambia, pero la relación ($X \to Y$) se mantiene.

> **CalmOps** usa detectores estadísticos (como Kolmogorov-Smirnov) para avisarte: *"¡Cuidado! Los usuarios están enviando datos muy diferentes a los que usaste para entrenar"*.

---

## 2. ¿Qué es Concept Drift? (La "Verdad" cambia)

Imagina un modelo para predecir precios de casas.
En 2020, una casa de 100m² costaba 100k€.
En 2025, la **misma** casa de 100m² cuesta 200k€ debido a la inflación.

Aquí los datos de entrada no cambiaron (la casa sigue midiendo 100m²), pero el **resultado esperado** sí. La "verdad" del mercado ha cambiado.

Esto es **Concept Drift**.
*   **En resumen:** La relación entre la entrada y la salida ($X \to Y$) cambia.

> **CalmOps** detecta esto monitoreando el error del modelo. Si el error sube repentinamente, es probable que haya ocurrido Concept Drift y necesites re-entrenar.

---

## 3. ¿Por qué Datos Sintéticos?

¿Por qué usar `SyntheticGenerator` o `RealGenerator` en lugar de datos reales?

1.  **Privacidad:** Puedes entrenar modelos con características estadísticas idénticas a los datos de pacientes reales, sin exponer la identidad de ningún paciente real.
2.  **Simulación de Crisis:** ¿Cómo se comportará tu IA si ocurre una pandemia mañana (cambio drástico de datos)? No puedes "esperar" a que ocurra.
    *   Con **CalmOps**, puedes usar `DriftInjector` para **crear** esa crisis artificialmente y ver si tu sistema de monitoreo la detecta.

---

## 4. Arquitectura Mental de CalmOps

Para usar la librería, visualiza tres actores:

1.  **El Generador (`Generator`)**:
    *   Tu "adversario". Crea datos, normales o corruptos/drifted.
    *   *Ejemplo:* `ClimateGenerator`, `SyntheticGenerator`.

2.  **El Monitor (`Monitor`)**:
    *   El "guardia de seguridad". Observa los datos que llegan.
    *   Si ve algo raro (drift), hace sonar la alarma.

3.  **El Pipeline (`Pipeline`)**:
    *   El "cerebro". Recibe la alarma, re-entrena el modelo automáticamente con los nuevos datos y restaura el orden.

---

## 🎓 Tu Camino de Aprendizaje

1.  Ejecuta `Synthetic/tutorial.py` para ver cómo se crean datos que cambian con el tiempo ("Drifts").
2.  Usa esos datos para alimentar un Pipeline simple.
3.  Observa en el **Dashboard** cómo las métricas de error suben cuando ocurre el drift y bajan cuando el Pipeline re-entrena.

**¡Bienvenido a MLOps!**
