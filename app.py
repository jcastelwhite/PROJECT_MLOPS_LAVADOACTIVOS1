# dashboard_la_ft.py

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

# ==========================
#  Cargar modelo ML
# ==========================
try:
    modelo = joblib.load("modelo_xgboost_laft.pkl")
    print("Modelo XGBoost cargado correctamente.")
except Exception as e:
    print("No se pudo cargar el modelo, usando modo simulado.", e)
    modelo = None

# ==========================
#  Funciones auxiliares
# ==========================

def generar_datos_mock(n=200):
    np.random.seed(42)
    data = pd.DataFrame({
        'id': range(1, n + 1),
        'monto': np.random.randint(10000, 5000000, n),
        'pais': np.random.choice(['Colombia', 'Panamá', 'EEUU', 'España', 'México'], n),
        'step': np.random.randint(1, 100, n),
        'oldbalanceOrg': np.random.randint(0, 5000000, n),
        'newbalanceOrig': np.random.randint(0, 5000000, n),
        'oldbalanceDest': np.random.randint(0, 5000000, n),
        'newbalanceDest': np.random.randint(0, 5000000, n),
    })
    return data

def predecir_riesgo_ml(data):
    """
    Aplica el modelo ML si está disponible.
    Debes adaptar las columnas de entrada según tu notebook (eda.ipynb)
    """
    if modelo is None:
        # Genera una columna simulada
        data['riesgo'] = np.random.choice(['Bajo', 'Medio', 'Alto'], len(data), p=[0.6, 0.25, 0.15])
    else:
        # Reemplaza esta parte con las columnas reales que usa tu modelo
        X = data[['step', 'monto', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
        y_pred = modelo.predict(X)

        # Si el modelo devuelve 0/1, convierte a etiquetas
        data['riesgo'] = np.where(y_pred == 1, 'Alto', 'Bajo')

    data['sospechosa'] = data['riesgo'] == 'Alto'
    return data

def calcular_estadisticas(data):
    total = len(data)
    total_monto = data['monto'].sum()
    sospechosas = data[data['sospechosa']]
    return {
        "total": total,
        "total_monto": total_monto,
        "sospechosas": len(sospechosas),
        "monto_sospechoso": sospechosas['monto'].sum(),
        "promedio": total_monto / total,
        "porcentaje_sospechosas": (len(sospechosas) / total) * 100
    }

# ==========================
#  Inicialización de app
# ==========================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Sistema de Detección LA/FT"

# ==========================
#  Layout principal
# ==========================
app.layout = dbc.Container([
    html.Br(),
    html.H2("🛡️ Sistema de Detección de Lavado de Activos y Financiación del Terrorismo",
            className="text-center mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Button("Cargar CSV", id="btn-cargar", color="primary", className="me-2"),
            dcc.Upload(
                id="upload-data",
                children=html.Div(["Arrastra o haz clic para subir un archivo CSV"]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            )
        ], width=12)
    ]),

    html.Hr(),

    dbc.Row(id="stat-cards", className="mb-4"),

    dbc.Row([
        dbc.Col([dcc.Graph(id="grafico-montos")], md=6),
        dbc.Col([dcc.Graph(id="grafico-riesgos")], md=6)
    ]),

    html.Br(),

    dbc.Row([
        dbc.Col([
            html.H4("📋 Transacciones"),
            dash_table.DataTable(
                id="tabla-transacciones",
                columns=[{"name": i, "id": i} for i in ['id','monto','pais','riesgo','sospechosa']],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'}
            )
        ], md=8),

        dbc.Col([
            html.H4("🚨 Alertas de Alto Riesgo"),
            html.Ul(id="lista-alertas", className="list-group")
        ], md=4)
    ])
], fluid=True)

# ==========================
#  Callbacks
# ==========================
@app.callback(
    Output("stat-cards", "children"),
    Output("grafico-montos", "figure"),
    Output("grafico-riesgos", "figure"),
    Output("tabla-transacciones", "data"),
    Output("lista-alertas", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def actualizar_tablero(contents, filename):
    import io, base64
    if contents and filename.endswith('.csv'):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    else:
        data = generar_datos_mock()

    # Aplicar modelo ML
    data = predecir_riesgo_ml(data)
    stats = calcular_estadisticas(data)

    # Tarjetas
    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Total Transacciones"), html.H2(f"{stats['total']:,}")]), color="light"), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Monto Total"), html.H2(f"${stats['total_monto']:,}")]), color="info", inverse=True), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Transacciones Sospechosas"), html.H2(f"{stats['sospechosas']:,}")]), color="danger", inverse=True), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Monto Sospechoso"), html.H2(f"${stats['monto_sospechoso']:,}")]), color="warning", inverse=True), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Promedio x Transacción"), html.H2(f"${stats['promedio']:.0f}")]), color="secondary", inverse=True), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Tasa Sospechosa"), html.H2(f"{stats['porcentaje_sospechosas']:.1f}%")]), color="success", inverse=True), md=2)
    ], className="g-2")

    # Gráficos
    fig_montos = px.histogram(data, x="monto", color="riesgo", nbins=20, title="Distribución de Montos por Nivel de Riesgo")
    fig_riesgos = px.pie(data, names="riesgo", title="Distribución de Riesgos")

    # Alertas
    alertas_html = [
        html.Li(f"Transacción #{row.id} - {row.pais} - ${row.monto:,}", className="list-group-item list-group-item-danger")
        for row in data[data['sospechosa']].itertuples()
    ]

    return cards, fig_montos, fig_riesgos, data.to_dict('records'), alertas_html

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=True)



