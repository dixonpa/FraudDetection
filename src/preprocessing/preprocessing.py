from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    """
    Crea el ColumnTransformer: escala las columnas numéricas y aplica one-hot a las categóricas.

    Se usa dentro de un Pipeline para que solo aprenda de los datos de entrenamiento.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
