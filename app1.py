

import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Detección de Fraude", layout="wide")
st.title("🕵️‍♂️ Sistema de Detección de Lavado de Activos y Financiación del Terrorismo")

# Cargar CSV
uploaded_file = st.file_uploader("📤 Cargar archivo de transacciones (.csv)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"{len(df)} transacciones cargadas.")
    
    # Mostrar tabla original
    with st.expander("📋 Ver transacciones originales"):
        st.dataframe(df.head(20))

    # Predecir con API Flask
    st.info("🔍 Enviando transacciones al modelo para predicción...")
    def predecir_fila(row):
        payload = {
            "type": row["type"],
            "amount": float(row["amount"]),
            "newbalanceDest": float(row["newbalanceDest"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
            "newbalanceOrig": float(row["newbalanceOrig"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "step": int(row["step"])
        }
        try:
            response = requests.post("http://127.0.0.1:5000/predict", json=payload)
            return response.json().get("prediction", 0)
        except:
            return 0

    df["prediccion"] = df.apply(predecir_fila, axis=1)

    # Métricas
    total = len(df)
    sospechosas = df["prediccion"].sum()
    monto_total = df["amount"].sum()
    monto_sospechoso = df[df["prediccion"] == 1]["amount"].sum()
    tasa = sospechosas / total * 100

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Transacciones", total)
    col2.metric("Monto Total", f"${monto_total:,.0f}")
    col3.metric("Sospechosas", sospechosas)
    col4.metric("Monto Sospechoso", f"${monto_sospechoso:,.0f}")
    col5.metric("Promedio", f"${monto_total/total:,.0f}")
    col6.metric("Tasa Sospechosa", f"{tasa:.1f}%")

    # Gráfica de montos
    fig = px.histogram(df, x="amount", color="prediccion",
                       title="Distribución de montos por predicción",
                       labels={"prediccion": "Sospechosa"})
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de alertas
    st.subheader("⚠️ Alertas de Alto Riesgo")
    alertas = df[df["prediccion"] == 1].sort_values(by="amount", ascending=False).head(10)
    st.dataframe(alertas)

    # Exportar resultados
    st.download_button("📥 Descargar resultados con predicción", df.to_csv(index=False), "resultados_fraude.csv", "text/csv")
