# Detección de Fraude en Transacciones Bancarias

## 📌 Descripción General

Este proyecto busca desarrollar un modelo de *machine learning* capaz de detectar transacciones fraudulentas en una entidad bancaria, superando el umbral de **0.75** en las métricas **AUC-PR**, **F1-score** y **F2-score**.  
Los datos utilizados provienen de un conjunto público de Kaggle y contienen información detallada sobre transacciones, clientes y comercios.

---

## 🎯 Objetivo

Predecir con alta precisión si una transacción es legítima o fraudulenta, minimizando tanto los **falsos negativos** como los **falsos positivos**, para proteger a los clientes y reducir pérdidas económicas por fraude.

---

## 📊 Tabla de Resultados

| Modelo                  | AUC-PR | F1-score | F2-score |
|-------------------------|--------|----------|----------|
| Random Forest           | 0.8767 | 0.8296   | 0.8067   |
| Random Forest con SMOTE | 0.8563 | 0.7539   | 0.8000   |
| XGBoost                 | 0.8698 | 0.3417   | 0.5574   |

---

## 🗂️ Estructura del Repositorio

```
/data/
  /raw/               # Datos originales sin procesar
  /processed/         # Datos después de limpieza y transformación

/notebooks/
  01_EDA.ipynb                # Análisis exploratorio de datos
  02_model_training.ipynb      # Preprocesamiento, transformación y modelado.
  03_results.ipynb           # Resultados obtenidos
  04_proyect_complete.ipynb         # Flujo del proyecto completo

/src/
  /data/
    data_clean.py # Script para la limpieza basica
  /features/
    engineering.py # Script para la creacion de variable de valor
  /modeling/
    models.py  # Script con 3 modelos, optimizacion y busqueda de hiperparametros
  /preprocessing/
    preprocessing.py  # Script para el escalado y codificacion de variables

/results/
  /figure/                   # Gráficos y visualizaciones
  /metrics/ # Metricas de cada modelo
  /transformers/  # Pipeline del flujo del trato de datos

README.md
requirements.txt
```

---

## 🔁 Resumen del Flujo de Trabajo

### 🔍 Exploración de Datos (EDA)
- Conversión de variables de fecha y hora.
- Análisis de duplicados y relaciones entre variables.
- Identificación de patrones temporales y grupos etarios más afectados por fraude.
- Detección de comercios con mayor concentración de fraude.

### 🛠️ Preprocesamiento
- Creación de variables como edad, hora, día de la semana y distancia geográfica.
- Eliminación de columnas irrelevantes.
- Normalización y codificación de variables categóricas.

### 🤖 Modelado
- Entrenamiento de Random Forest, Random Forest con SMOTE y XGBoost.
- Optimización de hiperparámetros mediante Random Search y Stratified K-Fold.
- Evaluación mediante AUC-PR, F1-score y F2-score.

### 📈 Resultados
- El modelo **Random Forest** fue el más robusto, superando el umbral de 0.75 en todas las métricas.
- **SMOTE** no mejoró el desempeño, posiblemente por ruido en el *oversampling*.
- **XGBoost** requiere mayor ajuste de hiperparámetros.

---

## 🧠 Principales Hallazgos

- Jóvenes (18–20 años) y adultos mayores (60+) tienen menos transacciones pero **mayor proporción de fraude**.
- Comercios como `'Shopping_net'` presentan **alta concentración de fraude**, pese a no ser los de mayor volumen.
- El modelo **Random Forest** es el más efectivo para este problema.

---

## ✅ Recomendaciones

- Implementar el modelo **Random Forest** en producción y actualizarlo periódicamente con nuevos datos.
- Priorizar la prevención bloqueando transacciones sospechosas, aunque ocasione molestias menores a clientes legítimos.
- Explorar otras técnicas de balanceo y ajustar hiperparámetros de **XGBoost** para futuros experimentos.

---

## 🚀 Cómo Reproducir el Proyecto

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/dixonpa/FraudDetection.git
   ```

2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar los notebooks en el siguiente orden desde la carpeta `/notebooks/`:
   - `01_EDA.ipynb`
   - `02_model_training.ipynb`
   - `03_results.ipynb`
   - `04_proyect_complete.ipynb` *(opcional para ver todo el flujo integral)*

---

## 👥 Créditos

- **Autor**: Paulo Alvarez.
- **Datos**: Kaggle

¿Tienes preguntas o sugerencias?  
¡No dudes en abrir un *issue* o contribuir al repositorio!

---
