import pandas as pd
from geopy.distance import geodesic

def calculate_age (df:pd.DataFrame, column_transaccion:str = "trans_date_trans_time", column_birth:str = "dob") -> pd.DataFrame:
    """
    Calcula la edad de los clientes a partir de su fecha de nacimiento y la fecha de transacción.

    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas de transacción y nacimiento
        column_transaccion (str): Nombre de la columna que contiene la fecha de transacción.
        column_birth (str): Nombre de la columna que contiene la fecha de nacimiento.
    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'edad' que contiene la edad de los clientes.
    """
    df['transaction_date'] = pd.to_datetime(df[column_transaccion])
    df['age'] = (df['transaction_date'] - df [column_birth]).dt.days // 365
    df.drop(columns=['transaction_date'], inplace=True)
    
    return df

def day_of_week(df: pd.DataFrame, column_transaccion: str = "trans_date_trans_time") -> pd.DataFrame:
    """
    Extrae el día de la semana de la fecha de transacción.
    Args:
        df (pd.DataFrame): DataFrame que contiene la columna de transacción.
        column_transaccion (str): Nombre de la columna que contiene la fecha de transacción.
    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'day_of_week' que contiene el día de la semana.
    """
    df['day_of_week'] = df[column_transaccion].dt.day_of_week

    return df

def hour_of_day(df: pd.DataFrame, column_transaccion: str = "trans_date_trans_time") -> pd.DataFrame:
    """
    Extrae la hora del día de la fecha de transacción.
    Args:
        df (pd.DataFrame): DataFrame que contiene la columna de transacción.
        column_transaccion (str): Nombre de la columna que contiene la fecha de transacción.
    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'hour_of_day' que contiene la hora del día.
    """
    df['hour_of_day'] = df[column_transaccion].dt.hour
    
    return df

def distance_transaction(df: pd.DataFrame, column_lat: str = "merch_lat", column_lon: str = "merch_long", column_user_lat: str = "lat", column_user_lon: str = "long") -> pd.DataFrame:
    """
    Calcula la distancia entre la ubicación del comerciante y la ubicación del usuario.

    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas de latitud y longitud.
        column_lat (str): Nombre de la columna que contiene la latitud del comerciante.
        column_lon (str): Nombre de la columna que contiene la longitud del comerciante.
        column_user_lat (str): Nombre de la columna que contiene la latitud del usuario.
        column_user_lon (str): Nombre de la columna que contiene la longitud del usuario.

    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'distance_transaction' que contiene la distancia entre el comerciante y el usuario.
    """

    # Calcula la distancia euclidiana entre las coordenadas del comerciante y del usuario
    # df['distance_transaction'] = ((df[column_lat] - df[column_user_lat]) ** 2 + (df[column_lon] - df[column_user_lon]) ** 2) ** 0.5

    df['distancia_km'] = df.apply(lambda row: geodesic((row['lat'], row['long']), 
                                                   (row['merch_lat'], row['merch_long'])).km, axis=1)

    return df