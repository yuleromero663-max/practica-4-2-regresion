
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==================================================
# PRACTICA 4.2 - REGRESION LINEAL Y CORRELACION
# PASO 1: SIMULACION DE DATOS EN DUCKDB
# ==================================================

# Crear conexion en memoria
con = duckdb.connect(database=":memory:")

# Establecer la semilla
con.execute("SELECT setseed(0.42);")

# Generar 100 registros sinteticos

# Generar variables predictoras y calcular CPC
# utilizando esas mismas variables

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

FROM datos_base;
""")


# Consultar la tabla
df_agua = con.execute(
    "SELECT * FROM telemetria_agua"
).df()

print("=== PASO 1: PRIMEROS 5 REGISTROS DE DUCKDB ===")
print(df_agua.head())

print("\nTotal de registros:", len(df_agua))

print("\nColumnas generadas:")
print(df_agua.columns.tolist())


# ==================================================
# PASO 2: MATRIZ DE CORRELACION DE PEARSON
# ==================================================

columnas = [
    "temperatura_max",
    "humedad_max",
    "personas_familia",
    "consumo_bidon_pc",
    "cpc_consumo"
]

# Calcular la matriz de correlacion
matriz_correlacion = df_agua[columnas].corr(
    method="pearson"
)

print("\n" + "=" * 65)
print("=== PASO 2: MATRIZ DE CORRELACION DE PEARSON ===")
print("=" * 65)

print(matriz_correlacion.round(4).to_string())

# Correlacion con la variable dependiente
print("\nCORRELACION CON EL CONSUMO DE AGUA:")

correlaciones = matriz_correlacion[
    "cpc_consumo"
].drop("cpc_consumo")

for variable, valor in correlaciones.items():
    print(f"{variable}: {valor:.4f}")

# Identificar la relacion mas fuerte
variable_mayor = correlaciones.abs().idxmax()

print("\nVARIABLE CON MAYOR CORRELACION ABSOLUTA:")
print(variable_mayor)

print(
    "Coeficiente r:",
    round(correlaciones[variable_mayor], 4)
)


# ==================================================
# PASO 3: REGRESION LINEAL MULTIPLE CON STATSMODELS
# ==================================================

# Variables independientes
X = df_agua[
    [
        "temperatura_max",
        "humedad_max",
        "personas_familia",
        "consumo_bidon_pc"
    ]
]

# Variable dependiente
Y = df_agua["cpc_consumo"]

# Agregar la constante del modelo
X_const = sm.add_constant(X)

# Ajustar el modelo de regresion OLS
modelo_ols = sm.OLS(Y, X_const).fit()

print("\n" + "=" * 65)
print("=== PASO 3: RESUMEN DEL MODELO OLS ===")
print("=" * 65)

# Mostrar tabla estadistica
print(modelo_ols.summary())

# Coeficientes de regresion
print("\n=== COEFICIENTES DEL MODELO ===")

for variable, coeficiente in modelo_ols.params.items():
    print(f"{variable}: {coeficiente:.4f}")

# Mostrar indicadores principales
print("\n=== INDICADORES DEL MODELO ===")

print(f"R2: {modelo_ols.rsquared:.4f}")

print(
    f"R2 ajustado: "
    f"{modelo_ols.rsquared_adj:.4f}"
)

print(
    f"Valor p del modelo: "
    f"{modelo_ols.f_pvalue:.6e}"
)

print("\n" + "=" * 65)


# ==================================================
# PASO 4: DIAGNOSTICO ESTADISTICO DE RESIDUOS
# ==================================================

print("\n" + "=" * 65)
print("=== PASO 4: DIAGNOSTICO DE RESIDUOS ===")
print("=" * 65)

# Obtener residuos del modelo OLS
residuos = modelo_ols.resid

# Prueba de normalidad Shapiro-Wilk
shapiro_stat, shapiro_p = stats.shapiro(residuos)

print("\nPRUEBA DE NORMALIDAD SHAPIRO-WILK")
print(f"Estadistico W: {shapiro_stat:.4f}")
print(f"Valor p: {shapiro_p:.6f}")

if shapiro_p > 0.05:
    print(
        "Decision: No se rechaza la hipotesis "
        "de normalidad de los residuos."
    )
else:
    print(
        "Decision: Se rechaza la hipotesis "
        "de normalidad de los residuos."
    )

# Estadistico Durbin-Watson
dw = durbin_watson(residuos)

print("\nPRUEBA DURBIN-WATSON")
print(f"Estadistico DW: {dw:.4f}")

if 1.5 <= dw <= 2.5:
    print(
        "Interpretacion orientativa: No se observa "
        "autocorrelacion de primer orden importante."
    )
else:
    print(
        "Interpretacion orientativa: Es recomendable "
        "revisar la posible autocorrelacion."
    )

print("\n" + "=" * 65)
print("DIAGNOSTICO ESTADISTICO FINALIZADO")
print("=" * 65)


# ==================================================
# PASO 5: EVALUACION PREDICTIVA CON SCIKIT-LEARN
# ==================================================

from sklearn.model_selection import train_test_split

print("\n" + "=" * 65)
print("=== PASO 5: EVALUACION PREDICTIVA ===")
print("=" * 65)

# Dividir datos: 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

# Crear y entrenar el modelo
modelo_sklearn = LinearRegression()
modelo_sklearn.fit(X_train, y_train)

# Realizar predicciones sobre datos no utilizados
# durante el entrenamiento
y_pred = modelo_sklearn.predict(X_test)

# Calcular indicadores predictivos
r2_prueba = r2_score(y_test, y_pred)

rmse_prueba = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

print("\nDISTRIBUCION DE DATOS")
print(f"Registros de entrenamiento: {len(X_train)}")
print(f"Registros de prueba: {len(X_test)}")

print("\nMETRICAS PREDICTIVAS")
print(f"R2 de prueba: {r2_prueba:.4f}")
print(f"RMSE de prueba: {rmse_prueba:.4f}")

print("\nCOMPARACION ENTRE VALORES REALES Y PREDICHOS")

comparacion = pd.DataFrame({
    "Real": y_test.to_numpy(),
    "Prediccion": y_pred
})

comparacion["Error"] = (
    comparacion["Real"] - comparacion["Prediccion"]
)

print(comparacion.head(10).round(4).to_string(index=False))

print("\n" + "=" * 65)
print("EVALUACION PREDICTIVA FINALIZADA")
print("=" * 65)


# ==================================================
# PASO 6: GENERACION DE GRAFICAS ESTADISTICAS
# ==================================================

import os
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 65)
print("=== PASO 6: GENERACION DE GRAFICAS ===")
print("=" * 65)

# Crear carpeta de salida junto al archivo Python
directorio_actual = os.path.dirname(
    os.path.abspath(__file__)
)

carpeta_graficas = os.path.join(
    directorio_actual, "graficas"
)

os.makedirs(carpeta_graficas, exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# GRAFICA 1: MATRIZ DE CORRELACION
# --------------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    matriz_correlacion,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.5
)

plt.title("Matriz de correlacion de Pearson")
plt.tight_layout()

plt.savefig(
    os.path.join(carpeta_graficas, "01_correlacion.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# GRAFICA 2: VALORES REALES VS PREDICHOS
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=y_test,
    y=y_pred,
    s=65
)

limite_min = min(y_test.min(), y_pred.min())
limite_max = max(y_test.max(), y_pred.max())

plt.plot(
    [limite_min, limite_max],
    [limite_min, limite_max],
    "r--",
    label="Prediccion perfecta"
)

plt.xlabel("Consumo real (L/hab/dia)")
plt.ylabel("Consumo predicho (L/hab/dia)")
plt.title("Valores reales vs. predichos")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(carpeta_graficas, "02_real_vs_predicho.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# GRAFICA 3: RESIDUOS VS VALORES AJUSTADOS
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=modelo_ols.fittedvalues,
    y=residuos,
    s=60
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Valores ajustados OLS")
plt.ylabel("Residuos")
plt.title("Diagnostico: residuos vs. valores ajustados")
plt.tight_layout()

plt.savefig(
    os.path.join(carpeta_graficas, "03_residuos.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# GRAFICA 4: Q-Q PLOT DE RESIDUOS
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

sm.qqplot(
    residuos,
    line="45",
    fit=True,
    ax=ax
)

ax.set_title("Grafica Q-Q de residuos")
plt.tight_layout()

plt.savefig(
    os.path.join(carpeta_graficas, "04_qq_plot.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# GRAFICA 5: HISTOGRAMA DE RESIDUOS
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.histplot(
    residuos,
    bins=15,
    kde=True
)

plt.axvline(
    x=0,
    color="red",
    linestyle="--",
    label="Error cero"
)

plt.xlabel("Residuo")
plt.ylabel("Frecuencia")
plt.title("Distribucion de los residuos")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(carpeta_graficas, "05_histograma_residuos.png"),
    dpi=300
)
plt.close()

print("\nSe generaron correctamente 5 graficas.")

print("\nArchivos creados:")

for archivo in sorted(os.listdir(carpeta_graficas)):
    if archivo.endswith(".png"):
        print("-", archivo)

print("\nUbicacion:")
print(carpeta_graficas)

print("\n" + "=" * 65)
print("GENERACION DE GRAFICAS FINALIZADA")
print("=" * 65)
