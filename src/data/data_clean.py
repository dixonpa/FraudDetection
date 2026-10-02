from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "kartik2112/fraud-detection"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

COLUMNS_TO_DROP = [
    "cc_num", "merchant", "first", "last", "street", "city", "state", "zip",
    "job", "trans_num", "unix_time", "lat", "long", "merch_lat", "merch_long", "dob",
]


def load_data(file_name: str) -> pd.DataFrame:
    """
    Carga un archivo del dataset (fraudTrain.csv o fraudTest.csv).

    Si el archivo no está en data/raw, lo descarga desde Kaggle con kagglehub.
    """
    path = DATA_DIR / file_name
    if not path.exists():
        import kagglehub  # solo se importa si hay que descargar

        print(f"{file_name} no está en data/raw, descargando desde Kaggle...")
        path = Path(kagglehub.dataset_download(KAGGLE_DATASET)) / file_name

    return pd.read_csv(path, index_col=0, parse_dates=["trans_date_trans_time", "dob"])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados y ordena las transacciones por fecha.
    """
    df = df.drop_duplicates(subset="trans_num")
    return df.sort_values("trans_date_trans_time").reset_index(drop=True)


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas que identifican al cliente o que ya se usaron para crear variables nuevas.
    """
    return df.drop(columns=COLUMNS_TO_DROP)
