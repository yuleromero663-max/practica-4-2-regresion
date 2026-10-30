
import streamlit as st
import duckdb
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

st.set_page_config(
    page_title="Practica 4.2 - Consumo de agua",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Analisis del consumo de agua potable")
st.caption(
    "Practica 4.2 | Regresion lineal y correlacion"
)

@st.cache_data
def cargar_datos():
    con = duckdb.connect(database=":memory:")
    con.execute("SELECT setseed(0.42)")

    con.execute("""
    CREATE TABLE telemetria_agua AS
    WITH datos_base AS (
        SELECT
            residencia_id,
            ROUND(18 + random() * 14, 2)
                AS temperatura_max,
            ROUND(50 + random() * 40, 2)
                AS humedad_max,
            ROUND(3 + random() * 4, 0)
                AS personas_familia,
            ROUND(0.2 + random() * 1.8, 2)
                AS consumo_bidon_pc,
            (random() * 6.0 - 3.0)
                AS error_aleatorio
        FROM generate_series(1, 100)
            AS t(residencia_id)
    )
    SELECT
        residencia_id,
        temperatura_max,
        humedad_max,
        personas_familia,
        consumo_bidon_pc,
        ROUND(
            94.0
            + 2.5 * temperatura_max
            - 0.8 * humedad_max
            - 5.2 * personas_familia
            + 15.0 * consumo_bidon_pc
            + error_aleatorio,
            2
        ) AS cpc_consumo
    FROM datos_base
    """)

    df = con.execute(
        "SELECT * FROM telemetria_agua"
    ).df()
    con.close()
    return df

df = cargar_datos()

variables = [
    "temperatura_max",
    "humedad_max",
    "personas_familia",
    "consumo_bidon_pc"
]

X = sm.add_constant(df[variables])
y = df["cpc_consumo"]
modelo = sm.OLS(y, X).fit()

st.subheader("Indicadores principales")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Residencias", len(df))
c2.metric("Consumo promedio",
          f"{df['cpc_consumo'].mean():.2f}")
c3.metric("R² OLS", f"{modelo.rsquared:.4f}")
c4.metric("R² ajustado",
          f"{modelo.rsquared_adj:.4f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Datos",
    "Correlaciones",
    "Graficas",
    "Prediccion"
])

with tab1:
    st.subheader("Datos simulados con DuckDB")
    st.dataframe(df, use_container_width=True)

    st.subheader("Coeficientes del modelo OLS")
    coeficientes = pd.DataFrame({
        "Variable": modelo.params.index,
        "Coeficiente": modelo.params.values,
        "Valor p": modelo.pvalues.values
    })
    st.dataframe(coeficientes, use_container_width=True)

    with st.expander("Resumen estadistico OLS"):
        st.code(modelo.summary().as_text())

with tab2:
    st.subheader("Matriz de correlacion de Pearson")
    matriz = df[variables + ["cpc_consumo"]].corr()
    st.dataframe(
        matriz.style.format("{:.4f}"),
        use_container_width=True
    )

with tab3:
    st.subheader("Graficas del analisis")

    carpeta = Path(__file__).resolve().parent / "graficas"

    archivos = [
        ("Matriz de correlacion", "01_correlacion.png"),
        ("Real vs. predicho", "02_real_vs_predicho.png"),
        ("Residuos", "03_residuos.png"),
        ("Q-Q plot", "04_qq_plot.png"),
        ("Histograma", "05_histograma_residuos.png")
    ]

    for titulo, nombre in archivos:
        ruta = carpeta / nombre

        if ruta.exists():
            st.markdown(f"#### {titulo}")
            st.image(str(ruta), use_container_width=True)
        else:
            st.warning(f"No se encontro {nombre}")

with tab4:
    st.subheader("Prediccion de consumo per capita")
    st.info(
        "Prediccion basada en datos simulados con fines academicos."
    )

    temperatura = st.number_input(
        "Temperatura maxima (°C)",
        min_value=0.0,
        value=25.0
    )

    humedad = st.number_input(
        "Humedad maxima (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    personas = st.number_input(
        "Personas por familia",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

    bidon = st.number_input(
        "Consumo de bidon per capita (L/hab/dia)",
        min_value=0.0,
        value=1.0
    )

    if st.button("Calcular consumo estimado",
                 type="primary"):

        entrada = pd.DataFrame([{
            "const": 1.0,
            "temperatura_max": temperatura,
            "humedad_max": humedad,
            "personas_familia": personas,
            "consumo_bidon_pc": bidon
        }])

        prediccion = modelo.predict(entrada).iloc[0]

        st.success(
            f"Consumo estimado: {prediccion:.2f} L/hab/dia"
        )

st.divider()
st.caption(
    "Proyecto integrador 4 - Analisis Estadistico Inferencial"
)
