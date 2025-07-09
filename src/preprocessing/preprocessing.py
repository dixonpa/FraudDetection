import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Tuple

def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Valida que el DataFrame contenga las columnas requeridas.

    Args:
        df (pd.DataFrame): DataFrame a validar
        required_cols (List[str]): Lista de nombres de columnas requeridas

    Raises:
        ValueError: Si alguna columna requerida no está presente
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan las siguientes columnas requeridas: {', '.join(missing_cols)}")

def create_preprocessor(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    """
    Crea un ColumnTransformer para el preprocesamiento de datos

    Args:
        categorical_cols (List[str]): Columnas categóricas
        numerical_cols (List[str]): Columnas numéricas

    Returns:
        ColumnTransformer: Transformador configurado para las columnas especificadas
    """
    # Definir los pasos de preprocesamiento para cada tipo de columna
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        drop='first',
        sparse_output=False
    )

    numerical_transformer = StandardScaler()

    # Crear el transformador de columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor
script_path = Path(__file__).parent
output_dir = script_path / "../results/transformers/"
def apply_preprocessing(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
    target_col: str,
    output_dir: str = output_dir
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Aplica el preprocesamiento usando Pipeline y ColumnTransformer

    Args:
        df_train (pd.DataFrame): Datos de entrenamiento
        df_test (pd.DataFrame): Datos de prueba
        categorical_cols (List[str]): Columnas categóricas
        numerical_cols (List[str]): Columnas numéricas
        target_col (str): Columna objetivo
        output_dir (str): Directorio para guardar transformadores

    Returns:
        Tuple: X_train, y_train, X_test, y_test preprocesados
        
    Raises:
        ValueError: Si hay problemas con los datos de entrada
    """
    try:
        # Validar columnas requeridas
        validate_columns(df_train, categorical_cols + numerical_cols + [target_col])
        validate_columns(df_test, categorical_cols + numerical_cols + [target_col])

        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Crear el preprocesador
        preprocessor = create_preprocessor(categorical_cols, numerical_cols)

        # Configurar el pipeline completo
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])

        # Ajustar y transformar los datos de entrenamiento
        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        
        full_pipeline.fit(X_train)
        
        # Transformar datos de entrenamiento y prueba
        X_train_preprocessed = full_pipeline.transform(X_train)
        X_test_preprocessed = full_pipeline.transform(df_test.drop(columns=[target_col]))
        
        # Obtener nombres de columnas después del preprocesamiento
        feature_names = preprocessor.get_feature_names_out()
        
        # Convertir a DataFrames con nombres de columnas adecuados
        X_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=feature_names, index=X_train.index)
        X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=feature_names, index=df_test.index)
        
        # Guardar el pipeline completo
        joblib.dump(full_pipeline, os.path.join(output_dir, "full_pipeline.pkl"))

        return X_train_preprocessed, y_train, X_test_preprocessed, df_test[target_col]

    except Exception as e:
        raise ValueError(f"Error en el preprocesamiento: {e}")