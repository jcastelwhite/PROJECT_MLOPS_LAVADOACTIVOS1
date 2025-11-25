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
    print("✅ Modelo XGBoost cargado correctamente.")
except Exception as e:
    print("⚠️ No se pudo cargar el modelo, usando modo simulado.", e)
    modelo = None

# ==========================
#  Funciones auxiliares
# ==========================

def generar_datos_mock(n=200):
    np.random.seed(42)
    data = pd.DataFrame({
        'id': range(1, n + 1),
        'type': np.random.choice(['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT', 'CASH_IN'], n),
        'amount': np.random.randint(1000, 5000000, n),
        'oldbalanceOrg': np.random.randint(0, 5000000, n),
        'newbalanceOrig': np.random.randint(0, 5000000, n),
        'oldbalanceDest': np.random.randint(0, 5000000, n),
        'newbalanceDest': np.random.randint(0, 5000000, n),
        'pais': np.random.choice(['Colombia', 'Panamá', 'EEUU', 'España', 'México'], n)
    })
    return data

def preparar_features(df):
    # Mismo preprocesamiento del modelo_prueba.py
    df = df.copy()
    percentage_fraud_type = {'CASH_OUT': 0.5, 'TRANSFER': 0.5, 'PAYMENT': 0, 'DEBIT': 0, 'CASH_IN': 0}
    df['type_percentage'] = df['type'].map(percentage_fraud_type)
    df['log_amount'] = np.log1p(df['amount'])
    df['newbalanceDestSqrt'] = np.sqrt(df['newbalanceDest'])
    df['vacied_account'] = np.where((df['type'] == 'TRANSFER') & (df['newbalanceDest'] == 0), 1, 0)
    epsilon = 1e-10
    df['balance_Change'] = (df['newbalanceDest'] - df['oldbalanceDest']) / (df['oldbalanceDest'] + epsilon)
    df['balance_Change_Origin'] = np.abs((df['newbalanceOrig'] - df['oldbalanceOrg']) / (df['oldbalanceOrg'] + epsilon))
    df['balance_Change'] = df['balance_Change'].clip(0, 100)
    df['balance_Change_Origin'] = df['balance_Change_Origin'].clip(0, 100)

    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, float('inf')]
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    df['balance_Change_Range'] = pd.cut(df['balance_Change'], bins=bins, labels=labels, include_lowest=True).astype(float)
    df['balance_Change_Origin_Range'] = pd.cut(df['balance_Change_Origin'], bins=bins, labels=labels, include_lowest=True).astype(float)
    return df

def predecir_riesgo_ml(df):
    if modelo is None:
        df['riesgo'] = np.random.choice(['Bajo', 'Alto'], len(df), p=[0.8, 0.2])
    else:
        df_proc = preparar_features(df)
        features = ['type_percentage', 'log_amount', 'vacied_account',
                    'balance_Change_Range', 'newbalanceDestSqrt', 'balance_Change_Origin_Range']
        X = df_proc[features].fillna(0)
        preds = modelo.predict(X)
        df['riesgo'] = np.where(preds == 1, 'Alto', 'Bajo')
    df['sospechosa'] = df['riesgo'] == 'Alto'
    return df

def calcular_estadisticas(df):
    total = len(df)
    total_monto = df['amount'].sum()
    sospechosas = df[df['sospechosa']]
    return {
        "total": total,
        "total_monto": total_monto,
        "sospechosas": len(sospechosas),
        "monto_sospechoso": sospechosas['amount'].sum(),
        "promedio": total_monto / total if total > 0 else 0,
        "porcentaje_sospechosas": (len(sospechosas) / total) * 100 if total > 0 else 0
    }

# ==========================
#  Inicialización de app
# ==========================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Sistema de Detección LA/FT"

app.layout = dbc.Container([
    html.Br(),
    html.H2("🛡️ Sistema de Detección de Lavado de Activos y Financiación del Terrorismo", className="text-center mb-4"),
    dcc.Upload(
        id="upload-data",
        children=html.Div(["📁 Arrastra o haz clic para subir un archivo CSV"]),
        style={'width': '100%', 'height': '60px','lineHeight': '60px','borderWidth': '1px','borderStyle': 'dashed',
               'borderRadius': '5px','textAlign': 'center','margin': '10px'},
        multiple=False
    ),
    html.Hr(),
    dbc.Row(id="stat-cards"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="grafico-montos"), md=6),
        dbc.Col(dcc.Graph(id="grafico-riesgos"), md=6)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H4("📋 Transacciones"),
            dash_table.DataTable(id="tabla-transacciones", page_size=10,
                                 style_table={'overflowX': 'auto'},
                                 style_cell={'textAlign': 'center'})
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
    Output("tabla-transacciones", "columns"),
    Output("lista-alertas", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def actualizar_tablero(contents, filename):
    import io, base64
    if contents and filename.endswith('.csv'):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    else:
        df = generar_datos_mock()

    df = predecir_riesgo_ml(df)
    stats = calcular_estadisticas(df)

    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Transacciones"), html.H3(f"{stats['total']:,}")]), color="light"), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Monto Total"), html.H3(f"${stats['total_monto']:,}")]), color="info", inverse=True), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Sospechosas"), html.H3(f"{stats['sospechosas']:,}")]), color="danger", inverse=True), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Monto Sospechoso"), html.H3(f"${stats['monto_sospechoso']:,}")]), color="warning", inverse=True), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Promedio x Tx"), html.H3(f"${stats['promedio']:.0f}")]), color="secondary", inverse=True), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Tasa Sospechosa"), html.H3(f"{stats['porcentaje_sospechosas']:.1f}%")]), color="success", inverse=True), md=2)
    ], className="g-2")

    fig_montos = px.histogram(df, x="amount", color="riesgo", nbins=20, title="Distribución de Montos por Nivel de Riesgo")
    fig_riesgos = px.pie(df, names="riesgo", title="Distribución de Riesgos")
    alertas_html = [html.Li(f"Tx #{row.id} - {row.pais} - ${row.amount:,}", className="list-group-item list-group-item-danger") for row in df[df['riesgo']=='Alto'].itertuples()]
    columns = [{"name": c, "id": c} for c in df.columns]
    return cards, fig_montos, fig_riesgos, df.to_dict('records'), columns, alertas_html

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=True)

#Este codigo fue generado con apoyo de IA
