[English](README.md) | **Español**

# Detección de fraude en transacciones con tarjeta

Proyecto de Machine Learning para detectar transacciones fraudulentas con tarjeta de crédito. La meta era lograr **AUC-PR, F1 y F2 mayores a 0.75**.

## Resultados

El mejor modelo fue **Random Forest** con un umbral de decisión de 0.2. Estos son sus resultados con los datos de prueba (junio a diciembre 2020):

| Métrica | Valor |
|---|---|
| AUC-PR | 0.88 |
| F1 | 0.82 |
| F2 | 0.84 |
| Recall | 0.85 |
| Precisión | 0.79 |

De 2145 fraudes, el modelo detecta 1817. A cambio, marca 481 transacciones normales como sospechosas, de más de 553 mil.

![Resultados en prueba](results/figures/resultados_prueba.png)

## Hallazgos principales

- El fraude se concentra entre las **22:00 y las 3:59**: en esas horas la tasa de fraude está entre 1.4% y 2.9%, y en el resto del día es de 0.1%.
- El monto mediano de un fraude es de 396 dólares, frente a 47 dólares en una transacción normal.
- Las compras por internet (`shopping_net`, `misc_net`) tienen la tasa de fraude más alta.
- Los mayores de 60 años tienen la tasa de fraude más alta.

![Fraude por hora](results/figures/fraude_por_hora.png)

## Datos

Dataset [Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) de Kaggle (datos simulados):

- `fraudTrain.csv`: 1.3 millones de transacciones (enero 2019 a junio 2020).
- `fraudTest.csv`: 556 mil transacciones (junio a diciembre 2020).

Solo el 0.58% de las transacciones son fraude.

Los archivos pesan más de 100 MB, así que no están en el repositorio. Si no están en `data/raw/`, el código los descarga automáticamente con `kagglehub`.

## Qué hice

1. **EDA** con los datos de entrenamiento: fraude por hora, día, edad, monto y categoría.
2. **Variables nuevas:** edad del cliente, hora, día de la semana y distancia entre el cliente y el comercio.
3. **Validación por fecha:** usé el último 20% del entrenamiento como validación, porque en fraude los datos tienen orden en el tiempo.
4. **Comparé** Dummy, regresión logística, Random Forest y XGBoost con AUC-PR, F1 y F2.
5. **Ajusté** Random Forest y elegí el umbral de decisión con los datos de validación.
6. **Evalué** el modelo final una sola vez con `fraudTest.csv`.

**Algo que corregí:** en la primera versión las gráficas por hora contaban todas las transacciones en lugar de los fraudes, y concluí que el fraude era de mediodía a medianoche. También hice el EDA con los datos de prueba, lo cual no es correcto.

## Estructura

```
FraudDetection/
├── data/raw/                 # datos de Kaggle (no se suben a GitHub)
├── notebooks/
│   ├── 01_eda.ipynb          # análisis exploratorio
│   └── 02_modelado.ipynb     # modelos, umbral y evaluación final
├── results/figures/          # gráficos
├── src/
│   ├── data/                 # carga y limpieza
│   ├── features/             # creación de variables
│   ├── preprocessing/        # escalado y one-hot encoding
│   └── modeling/             # métricas y umbral
└── requirements.txt
```

## Cómo ejecutarlo

```bash
git clone https://github.com/dixonpa/FraudDetection.git
cd FraudDetection
python -m venv .venv
.venv\Scripts\activate        # en Windows
source .venv/bin/activate     # en Mac/Linux
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

La primera vez, el notebook descarga los datos desde Kaggle (unos 200 MB). El notebook de modelado entrena varios modelos con más de un millón de filas y puede tardar 10 minutos o más.

## Herramientas

Python, pandas, scikit-learn, XGBoost, kagglehub, matplotlib, seaborn.

## Autor

Paulo Alvarez · [LinkedIn](https://www.linkedin.com/in/paulocealva) · [Portafolio](https://dixonpa.github.io/) · palvareza17@gmail.com
