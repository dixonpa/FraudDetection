import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, fbeta_score, precision_score, recall_score


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """
    Calcula las métricas del proyecto para un umbral de decisión.

    AUC-PR usa las probabilidades; el resto usa las predicciones con el umbral.
    """
    y_pred = (proba >= threshold).astype(int)
    return {
        "AUC-PR": average_precision_score(y_true, proba),
        "F1": f1_score(y_true, y_pred),
        "F2": fbeta_score(y_true, y_pred, beta=2),
        "Recall": recall_score(y_true, y_pred),
        "Precisión": precision_score(y_true, y_pred, zero_division=0),
    }


def scores_by_threshold(y_true, proba, thresholds=np.arange(0.1, 0.95, 0.05)) -> pd.DataFrame:
    """
    Calcula F1 y F2 para varios umbrales, para elegir el mejor.
    """
    rows = []
    for threshold in thresholds:
        y_pred = (proba >= threshold).astype(int)
        rows.append({
            "umbral": round(threshold, 2),
            "F1": f1_score(y_true, y_pred),
            "F2": fbeta_score(y_true, y_pred, beta=2),
        })
    return pd.DataFrame(rows).set_index("umbral")
