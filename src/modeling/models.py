import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, average_precision_score, fbeta_score, f1_score
from tqdm import tqdm
import sys

# Función para crear directorios si no existen
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✔ Directorio creado: {directory}")

def train_evaluate_model(pipeline, param_grid, X_train, y_train, X_test, y_test, model_name, n_iter=15, cv_splits=5):
    """
    Entrena y evalúa un modelo con búsqueda de hiperparámetros
    - n_iter: Número de iteraciones para RandomizedSearchCV
    - cv_splits: Número de folds para validación cruzada
    """
    print(f"\n🔧 Configuración para {model_name}:")
    print(f"   Muestra: {X_train.shape[0]} registros")
    print(f"   Iteraciones: {n_iter}")
    print(f"   Folds validación: {cv_splits}")
    print(f"   Parámetros a probar: {len(param_grid)} combinaciones")
    
    # Búsqueda de hiperparámetros
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='average_precision',
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=42,
        error_score='raise'
    )
    
    # Barra de progreso
    total_iterations = n_iter * cv_splits
    progress_desc = f"⏳ {model_name} ({X_train.shape[0]} muestras)"
    
    with tqdm(total=total_iterations, 
             desc=progress_desc, 
             unit="fold",
             file=sys.stdout,
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        
        # Callback para actualizar barra
        class Callback:
            def __init__(self, pbar):
                self.pbar = pbar
            def __call__(self, *args, **kwargs):
                self.pbar.update(1)
        
        search.callback = Callback(pbar)
        search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    
    # Evaluación
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Métricas clave - Añadimos F1-Score
    metrics = {
        'best_params': search.best_params_,
        'auc_pr': average_precision_score(y_test, y_proba),
        'f1_score': f1_score(y_test, y_pred),
        'f2_score': fbeta_score(y_test, y_pred, beta=2),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    # Guardar modelo y resultados
    ensure_dir('models')
    joblib.dump(best_model, f'../models/{model_name}.pkl')
    
    # Guardar métricas
    ensure_dir('results')
    metrics_df = pd.DataFrame({
        'metric': list(metrics.keys()),
        'value': [str(v) for v in metrics.values()]
    })
    metrics_df.to_csv(f'../results/{model_name}_metrics.csv', index=False)
    
    return best_model, metrics

def get_models_config(scale_pos_weight):
    """Define pipelines y espacios de parámetros para los modelos"""
    return {
        'rf': {
            'pipeline': RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        },
        'rf_smote': {
            'pipeline': ImbPipeline([
                ('smote', SMOTE(sampling_strategy=0.3, random_state=42)),
                ('classifier', RandomForestClassifier(
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            'param_grid': {
                'smote__k_neighbors': [3, 5, 7],
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        'xgb': {
            'pipeline': XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric='aucpr',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            ),
            'param_grid': {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 0.5]
            }
        }
    }

def run_all_models(X_train, y_train, X_test, y_test, test_mode=False):
    """Ejecuta y compara los modelos
    - test_mode: Si es True, reduce iteraciones y folds para pruebas rápidas
    """
    # Calcular desbalance
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    print(f"\n⚖️ Desbalance de clases: 1:{scale_pos_weight:.2f}")
    
    # Configuración según modo
    if test_mode:
        print("\n⚠️ MODO PRUEBA ACTIVADO (configuración rápida)")
        n_iter = 5      # Iteraciones reducidas
        cv_splits = 2   # Folds reducidos
    else:
        n_iter = 15     # Iteraciones completas
        cv_splits = 5   # Folds completos
    
    # Obtener configuraciones
    models_config = get_models_config(scale_pos_weight)
    
    results = {}
    
    # Crear directorios principales
    ensure_dir('models')
    ensure_dir('results')
    
    # Barra de progreso general
    model_names = list(models_config.keys())
    print(f"\n🚀 Comenzando entrenamiento de {len(model_names)} modelos")
    
    with tqdm(total=len(model_names), 
             desc="Progreso General", 
             unit="modelo",
             position=0,
             leave=True,
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as main_pbar:
        
        for model_name in model_names:
            config = models_config[model_name]
            
            print(f"\n{'='*50}")
            print(f"⚙️ Entrenando modelo: {model_name.upper()}")
            print(f"{'='*50}")
            
            model, metrics = train_evaluate_model(
                pipeline=config['pipeline'],
                param_grid=config['param_grid'],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                n_iter=n_iter,
                cv_splits=cv_splits
            )
            
            results[model_name] = metrics
            
            # Reporte intermedio - Añadimos F1-Score
            print(f"\n✅ Modelo {model_name} entrenado")
            print(f"📊 AUC-PR: {metrics['auc_pr']:.4f}")
            print(f"🎯 F1-Score: {metrics['f1_score']:.4f}")
            print(f"🚨 F2-Score: {metrics['f2_score']:.4f}")
            
            # Actualizar barra principal
            main_pbar.update(1)
            main_pbar.set_description(f"✅ {model_name.upper()} completado")
    
    return results