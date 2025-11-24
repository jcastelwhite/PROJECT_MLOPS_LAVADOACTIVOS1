import pandas as pd
import kagglehub
import os
import shutil # Aunque no se usará shutil.copyfile, lo mantengo por si acaso
import numpy as np

# import matplotlib.pyplot as plt # No se usan en este script
# import seaborn as sns # No se usan en este script
# from scipy.stats import skew, kurtosis # No se usan en este script

from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, recall_score # Solo las métricas que usas

from xgboost import XGBClassifier

# --- 1. Descarga y Carga de Datos (Simplificado) ---
path = kagglehub.dataset_download("stefanouccelli/tesis-1")
files_in_path = os.listdir(path)

# Construct the full path to the downloaded file
old_file_path = os.path.join(path, files_in_path[0])

# Construct the new file path in a writable directory
new_file_path_in_writable = os.path.join( files_in_path[0])

# Copy the file to the writable directory
shutil.copyfile(old_file_path, new_file_path_in_writable)

print(f"Copied '{old_file_path}' to '{new_file_path_in_writable}'")

# Construct the final file path after removing '.crdownload'
final_file_path = new_file_path_in_writable.replace('.crdownload', '')

# Rename the copied file
os.rename(new_file_path_in_writable, final_file_path)

print(f"Renamed '{new_file_path_in_writable}' to '{final_file_path}'")

# Load the renamed file into a pandas DataFrame
df = pd.read_csv(final_file_path)



# --- 2. Preprocesamiento y Feature Engineering ---
df = df[df['nameOrig'].notna()].copy() # Añadir .copy() para evitar SettingWithCopyWarning

# num_duplicated_rows no se usa, se puede eliminar si no es necesario
# duplicated_rows = df[df.duplicated(keep=False)]
# num_duplicated_rows = duplicated_rows.shape[0]

# Calcular porcentajes de fraude por tipo de transacción
percentage_fraud_type = df[df['isFraud']==1]['type'].value_counts(normalize=True)
dicc_percentage = {
'CASH_OUT': percentage_fraud_type.get('CASH_OUT', 0),
'TRANSFER': percentage_fraud_type.get('TRANSFER', 0),
'PAYMENT': percentage_fraud_type.get('PAYMENT', 0),
'DEBIT': percentage_fraud_type.get('DEBIT', 0),
'CASH_IN': percentage_fraud_type.get('CASH_IN', 0)
}

df['type_percentage'] = df['type'].map(dicc_percentage)
df['log_amount'] = np.log1p(df['amount']) # Ya estaba en tu código
df['newbalanceDestSqrt'] = np.sqrt(df['newbalanceDest'])

df['vacied_account'] = np.where( ( (df['type']=='TRANSFER') & (df['newbalanceDest']==0) ), 1, 0)

epsilon = 1e-10 # Usar epsilon para evitar división por cero
df['balance_Change'] = (df['newbalanceDest'] - df['oldbalanceDest']) / (df['oldbalanceDest'] + epsilon)
df['balance_Change_Origin'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) / (df['oldbalanceOrg'] + epsilon)
df['balance_Change_Origin'] = np.abs(df['balance_Change_Origin'])

# Asegura que balance_Change esté entre 0 y 100
df['balance_Change'] = df['balance_Change'].clip(0, 100)
df['balance_Change_Origin'] = df['balance_Change_Origin'].clip(0, 100)

bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, float('inf')]
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

df['balance_Change_Range'] = pd.cut(df['balance_Change'], bins=bins, labels=labels, include_lowest=True, right=False)
df['balance_Change_Origin_Range'] = pd.cut(df['balance_Change_Origin'], bins=bins, labels=labels, include_lowest=True, right=False)

# Manejar NaNs y convertir a int. Rellenar NaNs con 0 (o un valor que indique "fuera de rango")
#   df['balance_Change_Origin_Range'] = df['balance_Change_Origin_Range'].fillna(0)
#df['balance_Change_Origin_Range'] = df['balance_Change_Origin_Range'].astype(int)
#df['balance_Change_Range'] = df['balance_Change_Range'].fillna(0).astype(int)

#convertir en int
df['balance_Change_Range'] = df['balance_Change_Range'].astype(float)
df['balance_Change_Origin_Range'] = df['balance_Change_Origin_Range'].astype(float)

# --- 3. Preparación del Modelo ---
df1 = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
x = df1[['type_percentage', 'log_amount', 'vacied_account', 'balance_Change_Range', 'newbalanceDestSqrt','balance_Change_Origin_Range']]
y = df1['isFraud']

x.info()  # Verifica que no haya NaNs en las características

# --- 4. División de Datos ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, stratify=y)


mlflow.set_tracking_uri("file:./mlruns")

# --- 5. Entrenamiento del Modelo XGBoost con MLflow ---
experiment = mlflow.set_experiment("Fraude_Equipo")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Calcula scale_pos_weight para manejar el desbalance
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    scale_pos_weight_value = neg_count / pos_count

# Define los parámetros del modelo (corregidos)
# eval_metric='logloss' es un buen default, o 'aucpr' para desbalance
# max_features no es un parámetro de XGBoost, se elimina
# sample_weight se pasa a .fit(), no al constructor, se usa scale_pos_weight
    params = {
        'eval_metric': 'logloss', # Corregido: 'pre' no es válido. 'logloss' o 'aucpr' son buenas opciones.
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'scale_pos_weight': scale_pos_weight_value, # ¡Importante para datos desbalanceados!
        'use_label_encoder': False, # Para evitar un warning en versiones recientes de XGBoost
        'random_state': 123 # Para reproducibilidad
    }

    # Crea el modelo con los parámetros definidos y entrénalo
    xg = XGBClassifier(**params)
    xg.fit(x_train, y_train)

    # Realiza predicciones de prueba
    predictions = xg.predict(x_test)

    # Registra los parámetros
    mlflow.log_param("eval_metric", params['eval_metric'])
    mlflow.log_param("max_depth", params['max_depth'])
    mlflow.log_param("learning_rate", params['learning_rate'])
    mlflow.log_param("n_estimators", params['n_estimators'])
    mlflow.log_param("subsample", params['subsample'])
    mlflow.log_param("scale_pos_weight", params['scale_pos_weight'])

    # Registra el modelo
    mlflow.sklearn.log_model(xg, "xgboost-fraud-model1") # Nombre más apropiado

    # Crea y registra las métricas de interés
    f1score = f1_score(y_test, predictions)
    mlflow.log_metric("f1score", f1score)
    print(f"F1-Score: {f1score}")

    recall = recall_score(y_test, predictions)
    print(f"Recall: {recall}")
    mlflow.log_metric("recall", recall)

import joblib
joblib.dump(xg, "modelo_xgboost_laft.pkl")
print("Modelo guardado como modelo_xgboost_laft.pkl")
