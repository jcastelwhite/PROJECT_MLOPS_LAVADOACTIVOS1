from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Diccionario y parámetros usados en la transformación
dicc_percentage = {
    'CASH_OUT': 0.503023,
    'TRANSFER': 0.496977,
    'PAYMENT': 0,
    'DEBIT': 0,
    'CASH_IN': 0
}
epsilon = 1e-10
bins = [0.0, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, float('inf')]
labels = list(range(1, 20))

# Clase personalizada para ingeniería de características
class IngenieriaTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_t = X.copy()
        df_t['type_percentage'] = df_t['type'].map(dicc_percentage)
        df_t['amount'] = np.sqrt(df_t['amount'])
        df_t['log_amount'] = np.log1p(df_t['amount'])
        df_t['newbalanceDestSqrt'] = np.sqrt(df_t['newbalanceDest'])
        df_t['vacied_account'] = np.where(
            ((df_t['type'] == 'TRANSFER') & (df_t['newbalanceDest'] == 0)), 1, 0
        )
        df_t['balance_Change'] = np.abs((df_t['newbalanceDest'] - df_t['oldbalanceDest']) / (df_t['oldbalanceDest'] + epsilon)).clip(0, 100)
        df_t['balance_Change_Origin'] = np.abs((df_t['newbalanceOrig'] - df_t['oldbalanceOrg']) / (df_t['oldbalanceOrg'] + epsilon)).clip(0, 100)
        df_t['balance_Change_Range'] = pd.cut(df_t['balance_Change'], bins=bins, labels=labels, include_lowest=True, right=False).astype(int)
        df_t['balance_Change_Origin_Range'] = pd.cut(df_t['balance_Change_Origin'], bins=bins, labels=labels, include_lowest=True, right=False).astype(int)

        return df_t[['type_percentage', 'log_amount', 'vacied_account',
                     'balance_Change_Range', 'newbalanceDestSqrt',
                     'balance_Change_Origin_Range', 'step']]

# Cargar el modelo entrenado
model = joblib.load("fraud_pipeline.pkl")

# Inicializar Flask
app = Flask(__name__)

# HTML para formulario web
html_form = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Fraude Predictor</title>
</head>
<body>
  <h2>Formulario de predicción de fraude</h2>
  <form id="fraudForm">
    <label>Tipo:</label>
    <select name="type">
      <option value="TRANSFER">TRANSFER</option>
      <option value="CASH_OUT">CASH_OUT</option>
    </select><br><br>

    <label>Monto:</label>
    <input type="number" name="amount" required><br><br>

    <label>Saldo destino nuevo:</label>
    <input type="number" name="newbalanceDest" required><br><br>

    <label>Saldo destino anterior:</label>
    <input type="number" name="oldbalanceDest" required><br><br>

    <label>Saldo origen nuevo:</label>
    <input type="number" name="newbalanceOrig" required><br><br>

    <label>Saldo origen anterior:</label>
    <input type="number" name="oldbalanceOrg" required><br><br>

    <label>Step:</label>
    <input type="number" name="step" required><br><br>

    <button type="submit">Predecir</button>
  </form>

  <h3 id="result"></h3>

  <script>
    document.getElementById("fraudForm").addEventListener("submit", function(e) {
      e.preventDefault();
      const formData = new FormData(e.target);
      const data = Object.fromEntries(formData.entries());

      data.amount = parseFloat(data.amount);
      data.newbalanceDest = parseFloat(data.newbalanceDest);
      data.oldbalanceDest = parseFloat(data.oldbalanceDest);
      data.newbalanceOrig = parseFloat(data.newbalanceOrig);
      data.oldbalanceOrg = parseFloat(data.oldbalanceOrg);
      data.step = parseInt(data.step);

      fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      })
      .then(res => res.json())
      .then(res => {
        if (res.error) {
          document.getElementById("result").innerText = "Error: " + res.error;
        } else {
          document.getElementById("result").innerText =
            `Predicción: ${res.prediction} | Probabilidad: ${res.probability.toFixed(4)}`;
        }
      });
    });
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return "API de detección de fraude funcionando 🚀"

@app.route("/form", methods=["GET"])
def form():
    return render_template_string(html_form)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])
        y_proba = model.predict_proba(df_input)[:, 1]
        y_pred = (y_proba > 0.4).astype(int)

        return jsonify({
            "prediction": int(y_pred[0]),
            "probability": float(y_proba[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)