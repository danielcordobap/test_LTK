import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score
)
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix





def preprocess(df):
    """Simple preprocessing pipeline"""
    df_processed = df.copy()
    
    # Encode categorical: product_category
    le_category = LabelEncoder()
    df_processed['product_category_encoded'] = le_category.fit_transform(
        df_processed['product_category']
    )
    
    # Handle missing sizes (Fashion items only have sizes)
    if df_processed['size_purchased'].notna().any():
        most_common_size = df_processed['size_purchased'].mode()[0]
        df_processed['size_purchased'].fillna(most_common_size, inplace=True)
        
        le_size = LabelEncoder()
        df_processed['size_encoded'] = le_size.fit_transform(
            df_processed['size_purchased']
        )
    
    # Feature selection
    feature_cols = [
        'customer_age', 'customer_tenure_days', 'product_category_encoded',
        'product_price', 'days_since_last_purchase', 'previous_returns',
        'product_rating', 'size_encoded', 'discount_applied'
    ]
    
    X = df_processed[feature_cols]
    y = df_processed['is_return']
    
    return X, y


def calcular_impacto_financiero(y_true, y_pred):
    """
    Proyecta los costos para un lote de 100,000 unidades
    con una tasa real de retorno del 22%.
    """
    # 1. Obtener tasas reales del modelo (Performance en Test)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    tpr = tp / (tp + fn) # Sensibilidad (Qué % de las devoluciones reales atrapamos)
    fpr = fp / (fp + tn) # Tasa de Falsos Positivos (Qué % de las ventas buenas molestamos)
    
    # 2. Definir Escenario de Negocio (100k unidades)
    LOTE_TOTAL = 100000
    TASA_REAL_RETORNO = 0.22
    
    n_devoluciones_reales = LOTE_TOTAL * TASA_REAL_RETORNO       # 22,000
    n_ventas_buenas = LOTE_TOTAL * (1 - TASA_REAL_RETORNO)       # 78,000
    
    # 3. Proyectar conteos basados en el TPR/FPR del modelo
    # De las 22,000 devoluciones, ¿cuántas atrapamos (TP) y cuántas se escapan (FN)?
    proj_tp = n_devoluciones_reales * tpr
    proj_fn = n_devoluciones_reales * (1 - tpr)
    
    # De las 78,000 ventas buenas, ¿cuántas molestamos (FP) y cuántas dejamos pasar (TN)?
    proj_fp = n_ventas_buenas * fpr
    proj_tn = n_ventas_buenas * (1 - fpr)
    
    # 4. Calcular Costos
    COSTO_INTERVENCION = 3   # Costo por revisar (TP y FP)
    COSTO_DEVOLUCION = 18    # Costo por no detectar (FN)
    
    gasto_revisiones = (proj_tp + proj_fp) * COSTO_INTERVENCION
    perdida_devoluciones = proj_fn * COSTO_DEVOLUCION
    costo_total_modelo = gasto_revisiones + perdida_devoluciones
    
    # Escenario Base (Sin modelo, te comes todas las devoluciones)
    costo_sin_modelo = n_devoluciones_reales * COSTO_DEVOLUCION
    ahorro = costo_sin_modelo - costo_total_modelo
    
    print("\n" + "="*60)
    print(f"ANÁLISIS FINANCIERO (Proyección 100k envíos / 22% Retornos)")
    print("="*60)
    print(f"Métricas del Modelo en Test:")
    print(f" -> Recall (Capacidad de detección): {tpr:.2%}")
    print(f" -> FPR (Tasa de falsa alarma):      {fpr:.2%}")
    print("-" * 60)
    print(f"Proyección de Costos:")
    print(f"1. Costo Operativo (Revisiones a $3):  ${gasto_revisiones:,.2f}")
    print(f"   (Se revisaron {int(proj_tp + proj_fp)} envíos)")
    print(f"2. Costo de Fugas (Devoluciones a $18): ${perdida_devoluciones:,.2f}")
    print(f"   (Se escaparon {int(proj_fn)} devoluciones)")
    print("-" * 60)
    print(f"COSTO TOTAL CON XGBOOST:  ${costo_total_modelo:,.2f}")
    print(f"COSTO SIN MODELO:         ${costo_sin_modelo:,.2f}")
    print(f"AHORRO NETO:              ${ahorro:,.2f}")
    if ahorro > 0:
        print(">> EL MODELO ES RENTABLE ✅")
    else:
        print(">> EL MODELO NO ES RENTABLE ❌")
    print("="*60);
