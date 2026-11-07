import io
import os
import time
import inspect
import pandas as pd
import plotly.express as px
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

st.set_page_config(page_title="Market Basket Analysis", page_icon="🛒", layout="wide")

RUTA_POR_DEFECTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Groceries_dataset.csv")
COLUMNAS_REQUERIDAS = {"Member_number", "Date", "itemDescription"}
COLUMNAS_VISTA = ["antecedente", "consecuente", "support", "confidence", "lift"]
MIN_SUPPORT_APRIORI = 0.003


# ----------------------------- Funciones de datos -----------------------------
@st.cache_data(show_spinner="Construyendo transacciones...")
def cargar_transacciones(csv_bytes: bytes):
    """Lee el CSV y agrupa por Member_number + Date para formar las transacciones."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    if not COLUMNAS_REQUERIDAS.issubset(df.columns):
        return None, None
    df = df.dropna()
    df["itemDescription"] = df["itemDescription"].astype(str).str.strip().str.lower()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).drop_duplicates(subset=["Member_number", "Date", "itemDescription"])
    transacciones = (df.groupby(["Member_number", "Date"])["itemDescription"]
                       .apply(lambda s: sorted(set(s))).tolist())
    return transacciones, len(df)


@st.cache_data(show_spinner="Codificando (one-hot)...", max_entries=2)
def codificar(transacciones):
    te = TransactionEncoder()
    matriz = te.fit(transacciones).transform(transacciones)
    return pd.DataFrame(matriz, columns=te.columns_)


@st.cache_data(show_spinner="Minando reglas...", max_entries=5)
def minar_reglas(matriz, algoritmo, min_support, max_len, min_conf, min_lift):
    funcion = apriori if algoritmo == "Apriori" else fpgrowth
    t0 = time.perf_counter()
    itemsets = funcion(matriz, min_support=min_support, use_colnames=True, max_len=max_len)
    tiempo = time.perf_counter() - t0
    if itemsets.empty or itemsets["itemsets"].apply(len).max() < 2:
        return pd.DataFrame(), len(itemsets), tiempo
    kwargs = dict(metric="confidence", min_threshold=min_conf)
    if "num_itemsets" in inspect.signature(association_rules).parameters:
        kwargs["num_itemsets"] = len(matriz)
    try:
        reglas = association_rules(itemsets, **kwargs)
    except ValueError:
        return pd.DataFrame(), len(itemsets), tiempo
    reglas = reglas[reglas["lift"] >= min_lift].reset_index(drop=True)
    if reglas.empty:
        return reglas, len(itemsets), tiempo
    reglas["antecedente"] = reglas["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    reglas["consecuente"] = reglas["consequents"].apply(lambda s: ", ".join(sorted(s)))
    reglas["regla"] = reglas["antecedente"] + " → " + reglas["consecuente"]
    return reglas.sort_values("lift", ascending=False).reset_index(drop=True), len(itemsets), tiempo


def interpretar_lift(lift):
    if lift >= 3:
        return f"Asociación muy fuerte ({lift:.1f}x)"
    if lift >= 1.5:
        return f"Asociación fuerte ({lift:.1f}x)"
    if lift > 1.05:
        return f"Asociación moderada ({lift:.2f}x)"
    return "Casi independiente"


def recomendar(productos, reglas, top_n=5, ordenar_por="lift"):
    entrada = {p.strip().lower() for p in productos}
    aplicables = reglas[reglas["antecedents"].apply(lambda a: a.issubset(entrada))
                        & reglas["consequents"].apply(lambda c: c.isdisjoint(entrada))]
    if aplicables.empty:
        return pd.DataFrame()
    res = aplicables.sort_values([ordenar_por, "confidence"], ascending=False).head(top_n).copy()
    res["interpretación"] = res["lift"].apply(interpretar_lift)
    return res[COLUMNAS_VISTA + ["interpretación"]].reset_index(drop=True)


# ----------------------------- Barra lateral -----------------------------
st.title("🛒 Market Basket Analysis: reglas de asociación")
st.caption("Proyecto de Minería de Datos · Groceries Dataset · Apriori y FP-Growth")

with st.sidebar:
    st.header("📂 Datos")
    archivo = st.file_uploader("Cargar CSV (Member_number, Date, itemDescription)", type="csv")
    st.header("⚙️ Parámetros")
    algoritmo = st.selectbox("Algoritmo", ["FP-Growth", "Apriori"])
    min_support = st.slider("min_support", 0.001, 0.05, 0.001, 0.0005, format="%.4f")
    if algoritmo == "Apriori" and min_support < MIN_SUPPORT_APRIORI:
        st.warning(f"Apriori consume mucha memoria con soportes bajos. Se aplica min_support = "
                   f"{MIN_SUPPORT_APRIORI}. Para valores menores usa FP-Growth.")
        min_support = MIN_SUPPORT_APRIORI
    min_conf = st.slider("min_confidence", 0.0, 1.0, 0.05, 0.01)
    min_lift = st.slider("min_lift", 1.0, 5.0, 1.0, 0.05)
    max_len = st.selectbox("Tamaño máximo del itemset", [2, 3], index=1)

if archivo is not None:
    csv_bytes = archivo.getvalue()
elif os.path.exists(RUTA_POR_DEFECTO):
    with open(RUTA_POR_DEFECTO, "rb") as f:
        csv_bytes = f.read()
else:
    st.info("Sube un CSV desde la barra lateral o incluye 'Groceries_dataset.csv' junto a app.py.")
    st.stop()

transacciones, n_filas = cargar_transacciones(csv_bytes)
if transacciones is None:
    st.error(f"El CSV debe contener las columnas: {sorted(COLUMNAS_REQUERIDAS)}")
    st.stop()

matriz = codificar(transacciones)
reglas, n_itemsets, tiempo = minar_reglas(matriz, algoritmo, min_support, max_len, min_conf, min_lift)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transacciones", f"{len(matriz):,}")
c2.metric("Productos únicos", f"{matriz.shape[1]:,}")
c3.metric("Itemsets frecuentes", f"{n_itemsets:,}")
c4.metric("Reglas", f"{len(reglas):,}")
c5.metric(f"Tiempo {algoritmo}", f"{tiempo:.3f} s")

tab_datos, tab_reglas, tab_graficos, tab_reco = st.tabs(
    ["📦 Datos", "📋 Reglas", "📊 Gráficos", "🛍️ Recomendador"])

# ----------------------------- Pestaña Datos -----------------------------
with tab_datos:
    frecuencias = matriz.sum().sort_values(ascending=False).head(15).reset_index()
    frecuencias.columns = ["producto", "transacciones"]
    fig = px.bar(frecuencias.iloc[::-1], x="transacciones", y="producto", orientation="h",
                 title="Top 15 productos más frecuentes", template="plotly_white")
    st.plotly_chart(fig)

if reglas.empty:
    for pestana in (tab_reglas, tab_graficos, tab_reco):
        with pestana:
            st.warning("No hay reglas con estos parámetros. Reduce min_support, "
                       "min_confidence o min_lift en la barra lateral.")
    st.stop()

# ----------------------------- Pestaña Reglas -----------------------------
with tab_reglas:
    f1, f2 = st.columns([2, 1])
    texto = f1.text_input("Filtrar reglas que contengan un producto (opcional)").strip().lower()
    orden = f2.selectbox("Ordenar por", ["lift", "confidence", "support"])
    vista = reglas if not texto else reglas[reglas["regla"].str.contains(texto, regex=False)]
    vista = vista.sort_values(orden, ascending=False)
    st.write(f"**{len(vista):,}** reglas mostradas")
    st.dataframe(vista[COLUMNAS_VISTA + ["leverage", "conviction"]].round(4))
    st.download_button("⬇️ Descargar reglas (CSV)",
                       vista[COLUMNAS_VISTA].to_csv(index=False).encode("utf-8"),
                       "reglas_asociacion.csv", "text/csv")

# ----------------------------- Pestaña Gráficos -----------------------------
with tab_graficos:
    fig_sc = px.scatter(reglas, x="support", y="confidence", color="lift", size="lift",
                        hover_name="regla", color_continuous_scale="Viridis", size_max=22,
                        title="Support vs. confidence (color y tamaño = lift)",
                        template="plotly_white")
    st.plotly_chart(fig_sc)
    top_n = st.slider("Top N reglas por lift", 5, 30, 15)
    top = reglas.nlargest(top_n, "lift").sort_values("lift")
    fig_bar = px.bar(top, x="lift", y="regla", orientation="h", color="confidence",
                     color_continuous_scale="Blues", title=f"Top {top_n} reglas por lift",
                     template="plotly_white", height=max(400, 28 * top_n))
    fig_bar.update_layout(yaxis_title=None)
    st.plotly_chart(fig_bar)
    if st.checkbox("Mostrar coordenadas paralelas"):
        fig_par = px.parallel_coordinates(reglas, dimensions=["support", "confidence", "lift"],
                                          color="lift", color_continuous_scale="Viridis")
        st.plotly_chart(fig_par)

# ----------------------------- Pestaña Recomendador -----------------------------
with tab_reco:
    st.subheader("Generador de combinaciones de productos")
    productos_con_reglas = sorted(set().union(*reglas["antecedents"]))
    elegidos = st.multiselect("Productos en la cesta", productos_con_reglas)
    n_reco = st.slider("Número de recomendaciones", 1, 15, 5)
    criterio = st.radio("Ordenar por", ["lift", "confidence", "support"], horizontal=True)
    if elegidos:
        resultado = recomendar(elegidos, reglas, n_reco, criterio)
        if resultado.empty:
            st.info("Sin reglas aplicables para esa cesta con los umbrales actuales.")
        else:
            st.dataframe(resultado.round(4))
            mejor = resultado.iloc[0]
            st.success(f"Mejor combinación: **{mejor['consecuente']}** "
                       f"(confidence {mejor['confidence']:.1%}, lift {mejor['lift']:.2f}).")
    else:
        st.info("Selecciona uno o más productos para ver las recomendaciones.")
