"""
================================================================================
   ROBOT DE ADMINISTRACIÓN DE PORTAFOLIOS CON SEÑALES CUANTITATIVAS
   Versión Streamlit (tema oscuro) — Proyecto final de Finanzas Computacionales
================================================================================
Misma esencia que la app de Quarto/Shiny, portada a Streamlit para web:
  1. Perfil del inversor       4. Asignación de activos
  2. Universo y señales        5. Backtest y validación
  3. Frontera eficiente

Reutiliza tal cual los módulos puros senales.py y portafolio.py.
Todo el estado pesado (descargas, frontera) se cachea con st.cache_data y se
conserva entre interacciones con st.session_state.
================================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import senales as S
import portafolio as P

# ------------------------------------------------------------------ Config UI
st.set_page_config(
    page_title="Robot de Portafolios con Señales Cuantitativas",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_TEMPLATE = "plotly_dark"

# Perfil -> coeficiente de aversión al riesgo (lambda) y color
PERFILES = {
    "Conservador": dict(lam=12.0, color="#0d9488"),
    "Moderado":    dict(lam=5.0,  color="#3b82f6"),
    "Agresivo":    dict(lam=1.5,  color="#ea580c"),
}

# Cuestionario: cada respuesta tiene un puntaje (escala 1-5)
Q1 = {"Más de 10 años": 5, "5 a 10 años": 4, "3 a 5 años": 3, "1 a 3 años": 2, "Menos de 1 año": 1}
Q2 = {"Invertiría más": 5, "Mantendría todo": 4, "Vendería una parte": 3, "Vendería todo": 1}
Q3 = {"Maximizar crecimiento": 5, "Crecimiento con algo de estabilidad": 3, "Preservar el capital": 1}
Q4 = {"Amplia (derivados, acciones)": 5, "Media (fondos, acciones)": 3, "Poca o ninguna": 1}


# ============================================================ Funciones cacheadas
@st.cache_data(show_spinner=False)
def cargar_datos(tickers, periodo):
    """Descarga OHLCV por activo. Devuelve close (DataFrame) y ohlc (dict)."""
    ohlc_dict, close_cols = {}, {}
    for t in tickers:
        try:
            d = yf.download(t, period=periodo, auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
            if len(d) > 50:
                ohlc_dict[t] = d
                close_cols[t] = d["close"]
        except Exception:
            continue
    close = pd.DataFrame(close_cols).dropna()
    return {"tickers": list(close.columns), "close": close, "ohlc": ohlc_dict}


@st.cache_data(show_spinner=False)
def frontera_cache(close, n_port):
    """Frontera eficiente simulada (cacheada por datos + n_port)."""
    mu, cov = P.estadisticos_anualizados(close)
    nube = P.frontera_simulada(mu, cov, n_port=int(n_port))
    return mu, cov, nube


# ============================================================ Estado del perfil
def calcular_perfil(score):
    if score <= 9:
        return "Conservador"
    elif score <= 15:
        return "Moderado"
    return "Agresivo"


# ====================================================================== SIDEBAR
st.sidebar.title("Controles")
pagina = st.sidebar.radio(
    "Sección",
    ["1 · Perfil del inversor", "2 · Universo y señales", "3 · Frontera eficiente",
     "4 · Asignación de activos", "5 · Backtest y validación"],
)

# --- Perfil (siempre disponible para todas las páginas) ---
with st.sidebar.expander("Perfil del inversor", expanded=(pagina.startswith("1"))):
    a1 = st.radio("1. Horizonte de inversión", list(Q1), index=2, key="q1")
    a2 = st.radio("2. Si su portafolio cayera 20% en un mes", list(Q2), index=2, key="q2")
    a3 = st.radio("3. Objetivo principal", list(Q3), index=1, key="q3")
    a4 = st.radio("4. Experiencia invirtiendo", list(Q4), index=1, key="q4")
    objetivo = st.slider("Monto objetivo a invertir (MXN)", 10000, 1000000, 100000, 10000, key="objetivo")

score = Q1[a1] + Q2[a2] + Q3[a3] + Q4[a4]
categoria = calcular_perfil(score)
lam = PERFILES[categoria]["lam"]
color_perfil = PERFILES[categoria]["color"]

# --- Universo y señales ---
with st.sidebar.expander("Universo y señales", expanded=(pagina.startswith("2"))):
    tickers_raw = st.text_area(
        "Tickers de Yahoo Finance (uno por línea o separados por coma)",
        value="GOOGL\nMSFT\nAAPL\nAMZN\nWALMEX.MX\nBIMBOA.MX", height=140, key="tickers")
    periodo = st.selectbox("Histórico a descargar",
                           ["1y", "2y", "3y", "5y"], index=2, key="periodo")
    senales_sel = st.multiselect("Señales cuantitativas (consenso AND)",
                                 list(S.CATALOGO_SENALES.keys()),
                                 default=["Filtro de tendencia (SMA 200)"], key="senales")
    if st.button("Descargar y analizar", type="primary", use_container_width=True):
        raw = tickers_raw.replace(",", "\n")
        tickers = list(dict.fromkeys(t.strip().upper() for t in raw.split("\n") if t.strip()))
        with st.spinner("Descargando precios de Yahoo Finance..."):
            st.session_state["datos"] = cargar_datos(tuple(tickers), periodo)

# --- Optimización ---
with st.sidebar.expander("Optimización", expanded=(pagina.startswith("3"))):
    n_port = st.slider("Portafolios aleatorios a simular", 1000, 10000, 4000, 500, key="n_port")

datos = st.session_state.get("datos")


def aviso_sin_datos():
    st.info("Primero presione **Descargar y analizar** en la barra lateral "
            "(sección *Universo y señales*).")


# ==================================================================== PÁGINA 1
def pagina_perfil():
    st.header("1 · Perfil del inversor")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"""
        <div style='text-align:center; padding:28px; border:2px solid {color_perfil};
                    border-radius:14px; background:{color_perfil}22;'>
            <div style='font-size:15px; color:#9ca3af;'>Su perfil es</div>
            <div style='font-size:44px; font-weight:800; color:{color_perfil};'>{categoria}</div>
            <div style='font-size:14px; color:#cbd5e1; margin-top:10px;'>
                Puntaje del cuestionario: <b>{score}</b> / 20<br>
                Coeficiente de aversión al riesgo (λ): <b>{lam}</b>
            </div>
        </div>""", unsafe_allow_html=True)
        st.caption("Responda el cuestionario en la barra lateral para recalibrar su perfil.")
    with c2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, number={"suffix": " / 20"},
            title={"text": f"Tolerancia al riesgo — {categoria}"},
            gauge={"axis": {"range": [4, 20]}, "bar": {"color": color_perfil},
                   "steps": [{"range": [4, 9], "color": "rgba(13,148,136,0.25)"},
                             {"range": [9, 15], "color": "rgba(59,130,246,0.25)"},
                             {"range": [15, 20], "color": "rgba(234,88,12,0.25)"}]}))
        fig.update_layout(height=320, template=PLOTLY_TEMPLATE,
                          paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)


# ==================================================================== PÁGINA 2
def pagina_universo():
    st.header("2 · Universo y señales")
    if datos is None or datos["close"].empty:
        aviso_sin_datos()
        return
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Métricas por activo (anualizadas)")
        tabla = P.metricas_activos(datos["close"]).copy()
        tabla["Rend. esperado"] = (tabla["Rend. esperado"] * 100).round(2).astype(str) + " %"
        tabla["Volatilidad"] = (tabla["Volatilidad"] * 100).round(2).astype(str) + " %"
        tabla["Sharpe"] = tabla["Sharpe"].round(3)
        st.dataframe(tabla, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Matriz de correlación")
        if datos["close"].shape[1] >= 2:
            corr = P.rendimientos_diarios(datos["close"]).corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, aspect="auto", template=PLOTLY_TEMPLATE)
            fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Se necesitan al menos 2 activos para la correlación.")
    st.caption(f"Señales activas: {', '.join(senales_sel) if senales_sel else '(ninguna)'}")


# =============================================================== Optimización
def calcular_optimizacion():
    """Calcula mu/cov/nube (cacheado) y los 3 portafolios notables con el λ actual."""
    if datos is None or datos["close"].empty or datos["close"].shape[1] < 2:
        return None
    mu, cov, nube = frontera_cache(datos["close"], n_port)
    return {"mu": mu, "cov": cov, "nube": nube,
            "w_minvar": P.optimo_min_varianza(mu, cov),
            "w_sharpe": P.optimo_max_sharpe(mu, cov),
            "w_perfil": P.optimo_por_aversion(mu, cov, lam)}


# ==================================================================== PÁGINA 3
def pagina_frontera():
    st.header("3 · Frontera eficiente")
    if datos is None or datos["close"].empty:
        aviso_sin_datos()
        return
    if datos["close"].shape[1] < 2:
        st.warning("Se requieren al menos 2 activos con datos para optimizar.")
        return
    opt = calcular_optimizacion()
    mu, cov, nube = opt["mu"], opt["cov"], opt["nube"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nube["volatilidad"], y=nube["rendimiento"], mode="markers",
        marker=dict(size=4, color=nube["sharpe"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Sharpe")),
        name="Portafolios", opacity=0.5))

    # Acciones individuales elegidas (vol = raíz de la diagonal de la covarianza)
    vol_activos = np.sqrt(np.diag(cov.values))
    fig.add_trace(go.Scatter(
        x=vol_activos, y=mu.values, mode="markers+text",
        marker=dict(size=11, color="#f9fafb", symbol="square", line=dict(width=1, color="#111")),
        text=list(mu.index), textposition="bottom center",
        textfont=dict(size=11, color="#f9fafb"), name="Acciones"))

    for key, nombre, col, sym in [
        ("w_minvar", "Mínima varianza", "#0d9488", "diamond"),
        ("w_sharpe", "Máximo Sharpe", "#ef4444", "star"),
        ("w_perfil", f"Su perfil ({categoria})", color_perfil, "circle")]:
        w = opt[key]
        r = P.rendimiento_portafolio(w.values, mu.values)
        v = P.volatilidad_portafolio(w.values, cov.values)
        fig.add_trace(go.Scatter(x=[v], y=[r], mode="markers+text",
            marker=dict(size=16, color=col, symbol=sym, line=dict(width=1, color="white")),
            text=[nombre], textposition="top center", name=nombre))

    fig.update_layout(height=520, template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                      title="Frontera eficiente (anualizada)", xaxis_title="Volatilidad",
                      yaxis_title="Rendimiento esperado")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparativo de portafolios notables")
    filas = [
        P.resumen_portafolio("Mínima varianza", opt["w_minvar"], mu, cov),
        P.resumen_portafolio("Máximo Sharpe", opt["w_sharpe"], mu, cov),
        P.resumen_portafolio(f"Su perfil ({categoria})", opt["w_perfil"], mu, cov),
    ]
    tabla = pd.DataFrame(filas)
    tabla["Rend. esperado"] = (tabla["Rend. esperado"] * 100).round(2).astype(str) + " %"
    tabla["Volatilidad"] = (tabla["Volatilidad"] * 100).round(2).astype(str) + " %"
    tabla["Sharpe"] = tabla["Sharpe"].round(3)
    st.dataframe(tabla, use_container_width=True, hide_index=True)


# ==================================================================== PÁGINA 4
def pagina_asignacion():
    st.header("4 · Asignación de activos")
    if datos is None or datos["close"].empty or datos["close"].shape[1] < 2:
        aviso_sin_datos()
        return
    opt = calcular_optimizacion()
    w = opt["w_perfil"]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader(f"Asignación recomendada — perfil {categoria}")
        wp = w[w > 0.001]
        fig = go.Figure(go.Pie(labels=wp.index, values=wp.values, hole=0.45,
                               textinfo="label+percent"))
        fig.update_layout(height=420, template=PLOTLY_TEMPLATE,
                          paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Montos y estado táctico")
        filas = []
        for activo in w.index:
            if w[activo] < 0.001:
                continue
            estado = "—"
            if activo in datos["ohlc"]:
                pos = S.posicion_combinada(datos["ohlc"][activo], senales_sel)
                estado = "Invertido" if int(pos.iloc[-1]) == 1 else "Efectivo"
            filas.append({"Activo": activo, "Peso": f"{w[activo]*100:.1f} %",
                          "Monto (MXN)": round(w[activo] * objetivo, 0),
                          "Estado señal": estado})
        df = pd.DataFrame(filas).sort_values("Peso", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Señales activas: {', '.join(senales_sel) if senales_sel else '(ninguna → siempre invertido)'}")


# =================================================================== Backtest
def calcular_backtest():
    opt = calcular_optimizacion()
    if opt is None:
        return None
    w = opt["w_perfil"]
    precios = datos["close"]
    rends = precios.pct_change().fillna(0)
    activos = [a for a in w.index if a in precios.columns]
    w_vec = w[activos] / w[activos].sum()

    ret_bh = (rends[activos] * w_vec.values).sum(axis=1)

    pos_matriz = pd.DataFrame(index=precios.index)
    for a in activos:
        if a in datos["ohlc"]:
            pos = S.posicion_combinada(datos["ohlc"][a], senales_sel)
            pos_matriz[a] = pos.reindex(precios.index).ffill().fillna(0)
        else:
            pos_matriz[a] = 1
    ret_overlay = (rends[activos] * w_vec.values * pos_matriz[activos].shift(1).fillna(0)).sum(axis=1)

    w_eq = np.repeat(1.0 / len(activos), len(activos))
    ret_eq = (rends[activos] * w_eq).sum(axis=1)

    base = pd.DataFrame({
        "Óptimo Buy & Hold": (1 + ret_bh).cumprod() * 100,
        "Óptimo + señales": (1 + ret_overlay).cumprod() * 100,
        "Equiponderado": (1 + ret_eq).cumprod() * 100,
    }, index=precios.index)
    rets = {"Óptimo Buy & Hold": ret_bh, "Óptimo + señales": ret_overlay, "Equiponderado": ret_eq}
    return {"base": base, "rets": rets}


# ==================================================================== PÁGINA 5
def pagina_backtest():
    st.header("5 · Backtest y validación")
    if datos is None or datos["close"].empty or datos["close"].shape[1] < 2:
        aviso_sin_datos()
        return
    bt = calcular_backtest()
    base, rets = bt["base"], bt["rets"]

    colores = {"Óptimo Buy & Hold": "#3b82f6", "Óptimo + señales": "#22c55e",
               "Equiponderado": "#9ca3af"}
    fig = go.Figure()
    for col in base.columns:
        fig.add_trace(go.Scatter(x=base.index, y=base[col], mode="lines", name=col,
                                 line=dict(color=colores[col],
                                           width=3 if col == "Óptimo + señales" else 2)))
    fig.update_layout(height=440, template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                      title="Crecimiento de $100 (base 100)", xaxis_title="Fecha",
                      yaxis_title="Valor", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Prueba SPA de Hansen (data snooping)
    p_spa = "—"
    try:
        from arch.bootstrap import SPA
        aligned = pd.DataFrame({"b": -rets["Óptimo Buy & Hold"],
                                "s": -rets["Óptimo + señales"]}).dropna()
        if len(aligned) > 30:
            spa = SPA(aligned["b"], aligned[["s"]], reps=500, seed=42)
            spa.compute()
            p_spa = f"{float(spa.pvalues['consistent']):.3f}"
    except Exception:
        p_spa = "n/d"

    st.subheader("Métricas de desempeño")
    filas = []
    for nombre, r in rets.items():
        r = r.dropna()
        cagr = (base[nombre].iloc[-1] / 100) ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        sharpe = (r.mean() * 252) / vol if vol > 0 else 0.0
        curva = base[nombre]
        max_dd = ((curva - curva.cummax()) / curva.cummax()).min()
        var95 = np.percentile(r, 5)
        filas.append({
            "Estrategia": nombre, "CAGR": f"{cagr*100:.2f} %",
            "Volatilidad": f"{vol*100:.2f} %", "Sharpe": f"{sharpe:.3f}",
            "Máx. Drawdown": f"{max_dd*100:.2f} %", "VaR 95% diario": f"{var95*100:.2f} %",
            "SPA p-value": p_spa if nombre == "Óptimo + señales" else "—",
        })
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
    st.caption("SPA p-value < 0.05 sugiere que el overlay de señales supera al Buy & Hold "
               "por algo más que azar.")


# ====================================================================== ROUTER
st.title("Robot de Administración de Portafolios con Señales Cuantitativas")

if pagina.startswith("1"):
    pagina_perfil()
elif pagina.startswith("2"):
    pagina_universo()
elif pagina.startswith("3"):
    pagina_frontera()
elif pagina.startswith("4"):
    pagina_asignacion()
else:
    pagina_backtest()
