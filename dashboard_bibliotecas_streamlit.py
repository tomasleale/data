
# -*- coding: utf-8 -*-
"""
Dashboard interactivo: Bibliotecas (Filiales + Regionales) con Streamlit + Plotly
---------------------------------------------------------------------------------
Ejecuta en consola:
    streamlit run dashboard_bibliotecas_streamlit.py

Requisitos (instalar una vez):
    pip install streamlit plotly pandas openpyxl numpy
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Dashboard Bibliotecas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# 1) CARGA DE ARCHIVOS (subida o autodetección en carpeta)
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

DEFAULT_FILIALES = "BD Bibliotecas Filiales 13-1 Completa y Depurada.xlsx"
DEFAULT_REGIONALES = "BD Bibliotecas Regionales 9-1-2025 Completa y Depurada.xlsx"

st.sidebar.header("Archivos de datos")
file_filiales = st.sidebar.file_uploader(
    "Cargar Excel de Filiales (.xlsx)",
    type=["xlsx"],
    key="filiales"
)
file_regionales = st.sidebar.file_uploader(
    "Cargar Excel de Regionales (.xlsx)",
    type=["xlsx"],
    key="regionales"
)

def _read_uploaded_or_default(uploaded, default_name: str) -> pd.DataFrame | None:
    """Lee un Excel desde un archivo subido o desde la carpeta local si existe."""
    if uploaded is not None:
        try:
            return pd.read_excel(uploaded, sheet_name="Sheet1")
        except ValueError:
            xls = pd.ExcelFile(uploaded)
            return pd.read_excel(uploaded, sheet_name=xls.sheet_names[0])
    else:
        default_path = HERE / default_name
        if default_path.exists():
            try:
                return pd.read_excel(default_path, sheet_name="Sheet1")
            except ValueError:
                xls = pd.ExcelFile(default_path)
                return pd.read_excel(default_path, sheet_name=xls.sheet_names[0])
        return None

@st.cache_data(show_spinner=False)
def cargar_datos(_filiales_buf, _regionales_buf) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_f = _read_uploaded_or_default(_filiales_buf, DEFAULT_FILIALES)
    df_r = _read_uploaded_or_default(_regionales_buf, DEFAULT_REGIONALES)
    return df_f, df_r

df_filiales, df_regionales = cargar_datos(file_filiales, file_regionales)

st.sidebar.caption("Tip: si no subes nada, el app intentará leer los archivos por nombre desde la misma carpeta del script.")

if df_filiales is None or df_regionales is None:
    st.warning(
        "Debes subir **ambos** archivos o colocarlos en la misma carpeta del script "
        f"con los nombres: `{DEFAULT_FILIALES}` y `{DEFAULT_REGIONALES}`."
    )
    st.stop()

# ---------------------------------------------------------------------
# 2) ARMONIZACIÓN Y LIMPIEZA
# ---------------------------------------------------------------------
df_filiales = df_filiales.copy()
df_regionales = df_regionales.copy()
df_filiales["tipo_biblioteca"] = "Filial"
df_regionales["tipo_biblioteca"] = "Regional"

cols_clave = [
    "region", "comuna", "bib_id", "dependencia",
    "funcionarios", "hardware", "software1", "ciberseguridad1",
    "dimension_capacidades_generales", "dimension_servicios",
    "dimension_infraestructura", "dimension_alianzas", "indicemadurez",
    "tipo_biblioteca"
]

def armonizar_columnas(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].copy()

df_filiales = armonizar_columnas(df_filiales, cols_clave)
df_regionales = armonizar_columnas(df_regionales, cols_clave)

df_all = pd.concat([df_filiales, df_regionales], ignore_index=True)

for c in ["region", "comuna", "dependencia", "bib_id", "tipo_biblioteca"]:
    df_all[c] = df_all[c].astype(str).str.strip()

num_cols = [
    "funcionarios", "hardware", "software1", "ciberseguridad1",
    "dimension_capacidades_generales", "dimension_servicios",
    "dimension_infraestructura", "dimension_alianzas", "indicemadurez"
]
for c in num_cols:
    df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

df_all["id_interno"] = (
    df_all["tipo_biblioteca"].str[:1] + "_" +
    df_all["region"].str.replace(r"\s+", "_", regex=True) + "_" +
    df_all["comuna"].str.replace(r"\s+", "_", regex=True) + "_" +
    df_all.groupby(["tipo_biblioteca","region","comuna"]).cumcount().astype(str)
)

display_cols = [
    "tipo_biblioteca", "region", "comuna", "bib_id", "dependencia",
    "indicemadurez", "dimension_capacidades_generales",
    "dimension_servicios", "dimension_infraestructura", "dimension_alianzas",
    "funcionarios", "hardware", "software1", "ciberseguridad1"
]

# ---------------------------------------------------------------------
# 3) SIDEBAR: FILTROS
# ---------------------------------------------------------------------
st.sidebar.header("Filtros")

regiones = sorted(df_all["region"].dropna().unique().tolist())
reg_sel = st.sidebar.multiselect("Región", options=regiones)

# Comunas dependientes de la región
df_tmp_comunas = df_all if not reg_sel else df_all[df_all["region"].isin(reg_sel)]
comunas = sorted(df_tmp_comunas["comuna"].dropna().unique().tolist())
com_sel = st.sidebar.multiselect("Comuna", options=comunas)

dependencias = sorted(df_all["dependencia"].dropna().unique().tolist())
dep_sel = st.sidebar.multiselect("Dependencia", options=dependencias)

tipos = sorted(df_all["tipo_biblioteca"].dropna().unique().tolist())
tipo_sel = st.sidebar.multiselect("Tipo de biblioteca", options=tipos, default=tipos)

def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    if reg_sel:
        f = f[f["region"].isin(reg_sel)]
    if com_sel:
        f = f[f["comuna"].isin(com_sel)]
    if dep_sel:
        f = f[f["dependencia"].isin(dep_sel)]
    if tipo_sel:
        f = f[f["tipo_biblioteca"].isin(tipo_sel)]
    return f

f = aplicar_filtros(df_all)

# ---------------------------------------------------------------------
# 4) ENCABEZADO E INDICADORES
# ---------------------------------------------------------------------
st.title("Dashboard de Bibliotecas — Streamlit + Plotly")
st.caption(f"Archivos cargados: Filiales ({len(df_filiales)}) y Regionales ({len(df_regionales)})")

c1, c2, c3, c4 = st.columns(4)
n_bibs = len(f)
prom_madurez = float(f["indicemadurez"].mean()) if n_bibs > 0 else np.nan
med_madurez = float(f["indicemadurez"].median()) if n_bibs > 0 else np.nan
std_madurez = float(f["indicemadurez"].std()) if n_bibs > 1 else np.nan

c1.metric("Bibliotecas (registros)", f"{n_bibs:,}".replace(",", "."))
c2.metric("Promedio índice madurez", "—" if np.isnan(prom_madurez) else f"{prom_madurez:,.2f}")
c3.metric("Mediana índice madurez", "—" if np.isnan(med_madurez) else f"{med_madurez:,.2f}")
c4.metric("Desv. estándar madurez", "—" if np.isnan(std_madurez) else f"{std_madurez:,.2f}")

st.divider()

# ---------------------------------------------------------------------
# 5) GRÁFICOS
# ---------------------------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("Distribución del índice de madurez")
    fig_hist = px.histogram(
        f, x="indicemadurez", nbins=20, marginal="box", opacity=0.9,
        title=None
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with colB:
    st.subheader("Promedio de madurez por región")
    reg_group = f.groupby("region", dropna=True, as_index=False)["indicemadurez"].mean()
    fig_bar_reg = px.bar(reg_group.sort_values("indicemadurez", ascending=False),
                         x="region", y="indicemadurez", title=None)
    fig_bar_reg.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar_reg, use_container_width=True)

st.subheader("Madurez por dependencia")
fig_box_dep = px.box(f, x="dependencia", y="indicemadurez", points="all", title=None)
fig_box_dep.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig_box_dep, use_container_width=True)

st.subheader("Relaciones entre dimensiones")
colX, colY = st.columns(2)
with colX:
    xvar = st.selectbox(
        "Eje X",
        options={
            "dimension_capacidades_generales":"Capacidades Generales",
            "dimension_servicios":"Servicios",
            "dimension_infraestructura":"Infraestructura",
            "dimension_alianzas":"Alianzas",
            "indicemadurez":"Indicador de Madurez",
        },
        index=0,
        format_func=lambda k: {
            "dimension_capacidades_generales":"Capacidades Generales",
            "dimension_servicios":"Servicios",
            "dimension_infraestructura":"Infraestructura",
            "dimension_alianzas":"Alianzas",
            "indicemadurez":"Indicador de Madurez",
        }[k]
    )
with colY:
    yvar = st.selectbox(
        "Eje Y",
        options={
            "indicemadurez":"Indicador de Madurez",
            "dimension_capacidades_generales":"Capacidades Generales",
            "dimension_servicios":"Servicios",
            "dimension_infraestructura":"Infraestructura",
            "dimension_alianzas":"Alianzas",
        },
        index=0,
        format_func=lambda k: {
            "dimension_capacidades_generales":"Capacidades Generales",
            "dimension_servicios":"Servicios",
            "dimension_infraestructura":"Infraestructura",
            "dimension_alianzas":"Alianzas",
            "indicemadurez":"Indicador de Madurez",
        }[k]
    )

scatter_df = f.copy()
for col in [xvar, yvar]:
    scatter_df[col] = pd.to_numeric(scatter_df[col], errors="coerce")
scatter_df = scatter_df.dropna(subset=[xvar, yvar])

fig_scatter = px.scatter(
    scatter_df,
    x=xvar, y=yvar, color="tipo_biblioteca",
    hover_data=["region","comuna","bib_id","dependencia","indicemadurez"],
    title=None
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------
# 6) RADAR POR BIBLIOTECA
# ---------------------------------------------------------------------
st.subheader("Detalle por biblioteca (Radar de dimensiones)")

def label_biblio(row: pd.Series) -> str:
    nombre = row.get("bib_id") or "Biblioteca"
    return f"{row['tipo_biblioteca']} • {row['region']} • {row['comuna']} • {nombre}"

opciones = f.apply(label_biblio, axis=1).tolist()
valores = f["id_interno"].tolist()
sel_id = None
if valores:
    sel_id = st.selectbox("Biblioteca", options=valores, format_func=lambda v: opciones[valores.index(v)])

def build_radar(row: pd.Series) -> go.Figure:
    dims = [
        ("Capacidades Generales", "dimension_capacidades_generales"),
        ("Servicios", "dimension_servicios"),
        ("Infraestructura", "dimension_infraestructura"),
        ("Alianzas", "dimension_alianzas"),
        ("Índice Madurez", "indicemadurez"),
    ]
    theta = [d[0] for d in dims]
    vals = [pd.to_numeric(row[d[1]], errors="coerce") for d in dims]

    vals_norm = []
    for name, v in zip(theta, vals):
        if pd.isna(v):
            vals_norm.append(np.nan)
        else:
            if name == "Índice Madurez" and v > 1.5:
                vals_norm.append(v/100.0)
            else:
                vals_norm.append(v)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_norm, theta=theta, fill="toself", name="Biblioteca"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(l=10,r=10,t=30,b=10),
        title=f"Radar: {row['bib_id']} ({row['comuna']}, {row['region']})"
    )
    return fig

if sel_id:
    fila = f.loc[f["id_interno"] == sel_id]
    if not fila.empty:
        fig_radar = build_radar(fila.iloc[0])
        st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Selecciona una biblioteca para ver su radar.")

# ---------------------------------------------------------------------
# 7) TABLA Y DESCARGA
# ---------------------------------------------------------------------
st.subheader("Tabla (registros filtrados)")
tabla = f[display_cols].copy()
for c in num_cols:
    if c in tabla.columns:
        tabla[c] = tabla[c].round(2)

st.dataframe(tabla, use_container_width=True, height=380)

csv_bytes = tabla.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar CSV filtrado",
    data=csv_bytes,
    file_name="bibliotecas_filtrado.csv",
    mime="text/csv"
)
