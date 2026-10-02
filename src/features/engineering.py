import numpy as np
import pandas as pd


def calculate_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la edad del cliente en el momento de la transacción.
    """
    df = df.copy()
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la hora del día y el día de la semana (0 = lunes) de la transacción.
    """
    df = df.copy()
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    return df


def calculate_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la distancia en km entre el cliente y el comercio con la fórmula de Haversine.

    Antes usaba geopy fila por fila, pero con más de un millón de filas era muy lento.
    """
    df = df.copy()
    lat1, lon1 = np.radians(df["lat"]), np.radians(df["long"])
    lat2, lon2 = np.radians(df["merch_lat"]), np.radians(df["merch_long"])

    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las transformaciones de feature engineering.
    """
    df = calculate_age(df)
    df = add_time_features(df)
    df = calculate_distance(df)
    return df
