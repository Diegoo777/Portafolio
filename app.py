"""
================================================================================
   APLICACION DE ADMINISTRACION DE PORTAFOLIOS Y ANALISIS TECNICO
   PEOYECTO INTEGRADOR
================================================================================

"""

import importlib.util

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# scikit-learn es opcional: sólo lo necesita la señal "Regresión logística (ML)".
SKLEARN_OK = importlib.util.find_spec("sklearn") is not None
SENAL_ML = "Regresión logística (ML)"

# Esquema de colores "spring" (magenta -> amarillo) para la nube de portafolios.
COLORSCALE_SPRING = [[0.0, "#ff00ff"], [0.25, "#ff40bf"], [0.5, "#ff8080"],
                     [0.75, "#ffbf40"], [1.0, "#ffff00"]]

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
     "4 · Asignación de activos", "5 · Backtest y validación", "6 · Indicadores técnicos"],
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
        value="QQQ\nBTC-USD\nMETA\nARKK\nTTWO\nNVDA", height=160, key="tickers")
    periodo = st.selectbox("Histórico a descargar",
                           ["1y", "2y", "3y", "5y"], index=2, key="periodo")
    senales_sel = st.multiselect("Señales cuantitativas",
                                 list(S.CATALOGO_SENALES.keys()),
                                 default=["Filtro de tendencia (SMA 200)"], key="senales")
    modo_combinacion = st.selectbox(
        "Cómo combinar varias señales",
        ["Consenso (AND)", "Cualquiera (OR)", "Mayoría (voto)"],
        index=0, key="modo_comb",
        help="AND: invertido sólo si TODAS coinciden (más tiempo en efectivo). "
             "OR: invertido si CUALQUIERA lo indica. "
             "Voto: invertido si la mayoría coincide.")
    if SENAL_ML in senales_sel and not SKLEARN_OK:
        st.warning("La señal **Regresión logística (ML)** requiere `scikit-learn`, que no "
                   "está instalado en este entorno: queda **desactivada** (no genera señales). "
                   "Instálalo con `pip install scikit-learn` o quítala de la selección.")
    if st.button("Descargar y analizar", type="primary", use_container_width=True):
        raw = tickers_raw.replace(",", "\n")
        tickers = list(dict.fromkeys(t.strip().upper() for t in raw.split("\n") if t.strip()))
        with st.spinner("Descargando precios de Yahoo Finance..."):
            st.session_state["datos"] = cargar_datos(tuple(tickers), periodo)

# --- Optimización ---
with st.sidebar.expander("Optimización", expanded=(pagina.startswith("3"))):
    n_port = st.slider("Portafolios aleatorios a simular", 1000, 10000, 4000, 500, key="n_port")

# --- Backtest ---
with st.sidebar.expander("Backtest", expanded=(pagina.startswith("5"))):
    rf_anual = st.slider("Rendimiento del efectivo (tasa libre de riesgo) % anual",
                         0.0, 12.0, 4.0, 0.5, key="rf",
                         help="Cuando una señal saca al activo del mercado, ese dinero "
                              "rinde esta tasa. En periodos de tasas altas favorece al overlay.") / 100.0
    costo_oper = st.slider("Costo por cambio de posición %",
                           0.0, 0.50, 0.0, 0.05, key="costo",
                           help="Comisión/deslizamiento aplicado cada vez que la señal entra "
                                "o sale del mercado. A más operaciones, más penaliza.") / 100.0

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
        marker=dict(size=4, color=nube["sharpe"], colorscale=COLORSCALE_SPRING,
                    showscale=True,
                    colorbar=dict(title="Sharpe", thickness=14, len=0.85, x=1.0, xpad=4)),
        name="Portafolios", opacity=0.55))

    # Acciones individuales elegidas (vol = raíz de la diagonal de la covarianza)
    vol_activos = np.sqrt(np.diag(cov.values))
    fig.add_trace(go.Scatter(
        x=vol_activos, y=mu.values, mode="markers+text",
        marker=dict(size=11, color="#f9fafb", symbol="square", line=dict(width=1, color="#111")),
        text=list(mu.index), textposition="bottom center",
        textfont=dict(size=11, color="#cbd5e1"), name="Acciones"))

    # Portafolios notables. Se alternan posiciones de etiqueta para que no se
    # encimen cuando dos óptimos caen casi en el mismo punto (p. ej. Máximo Sharpe
    # y el perfil del inversor), y el texto se colorea igual que su marcador.
    for key, nombre, col, sym, tpos in [
        ("w_minvar", "Mínima varianza", "#2dd4bf", "diamond", "middle left"),
        ("w_sharpe", "Máximo Sharpe", "#ef4444", "star", "top center"),
        ("w_perfil", f"Su perfil ({categoria})", color_perfil, "circle", "bottom center")]:
        w = opt[key]
        r = P.rendimiento_portafolio(w.values, mu.values)
        v = P.volatilidad_portafolio(w.values, cov.values)
        fig.add_trace(go.Scatter(x=[v], y=[r], mode="markers+text",
            marker=dict(size=16, color=col, symbol=sym, line=dict(width=1.5, color="white")),
            text=[nombre], textposition=tpos,
            textfont=dict(size=12, color=col), cliponaxis=False, name=nombre))

    fig.update_layout(height=560, template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                      title="Frontera eficiente (anualizada)", xaxis_title="Volatilidad",
                      yaxis_title="Rendimiento esperado",
                      margin=dict(t=70, r=95, l=60, b=90),
                      legend=dict(orientation="h", yanchor="top", y=-0.12,
                                  xanchor="center", x=0.5))
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
                pos = S.posicion_combinada(datos["ohlc"][activo], senales_sel, modo_combinacion)
                estado = "Invertido" if int(pos.iloc[-1]) == 1 else "Efectivo"
            filas.append({"Activo": activo, "Peso": f"{w[activo]*100:.1f} %",
                          "Monto (MXN)": round(w[activo] * objetivo, 0),
                          "Estado señal": estado})
        df = pd.DataFrame(filas).sort_values("Peso", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Señales activas: {', '.join(senales_sel) if senales_sel else '(ninguna → siempre invertido)'}")


# =================================================================== Backtest
def calcular_backtest(rf_anual=0.0, costo_oper=0.0, modo="Consenso (AND)"):
    opt = calcular_optimizacion()
    if opt is None:
        return None
    w = opt["w_perfil"]
    precios = datos["close"]
    rends = precios.pct_change().fillna(0)
    activos = [a for a in w.index if a in precios.columns]
    w_vec = w[activos] / w[activos].sum()
    rf_diario = (1 + rf_anual) ** (1 / 252) - 1

    # --- Óptimo Buy & Hold (siempre invertido) ---
    ret_bh = (rends[activos] * w_vec.values).sum(axis=1)

    # --- Óptimo + overlay de señales ---
    pos_matriz = pd.DataFrame(index=precios.index)
    for a in activos:
        if a in datos["ohlc"]:
            pos = S.posicion_combinada(datos["ohlc"][a], senales_sel, modo)
            pos_matriz[a] = pos.reindex(precios.index).ffill().fillna(1)
        else:
            pos_matriz[a] = 1
    # La posición se decide en t-1 y se ejecuta en t (sin look-ahead). El arranque
    # es invertido (1) para no arrastrar efectivo ocioso al inicio.
    pos_exec = pos_matriz[activos].shift(1).fillna(1)

    # Sleeve invertido -> retorno del activo; sleeve en efectivo -> tasa libre de riesgo.
    ret_por_activo = rends[activos] * pos_exec + (1 - pos_exec) * rf_diario
    ret_overlay = (ret_por_activo * w_vec.values).sum(axis=1)

    # Costos de transacción: se cobran cada vez que un activo cambia de posición.
    cambios = pos_exec.diff().abs().fillna(0)
    costo_diario = (cambios * w_vec.values).sum(axis=1) * costo_oper
    ret_overlay = ret_overlay - costo_diario

    # --- Equiponderado (referencia ingenua) ---
    w_eq = np.repeat(1.0 / len(activos), len(activos))
    ret_eq = (rends[activos] * w_eq).sum(axis=1)

    base = pd.DataFrame({
        "Óptimo Buy & Hold": (1 + ret_bh).cumprod() * 100,
        "Óptimo + señales": (1 + ret_overlay).cumprod() * 100,
        "Equiponderado": (1 + ret_eq).cumprod() * 100,
    }, index=precios.index)
    rets = {"Óptimo Buy & Hold": ret_bh, "Óptimo + señales": ret_overlay, "Equiponderado": ret_eq}
    exposicion = (pos_exec * w_vec.values).sum(axis=1)
    extra = {
        "pct_invertido": float(exposicion.mean()),
        "n_operaciones": int(cambios.sum().sum()),
        "rf_diario": rf_diario,
    }
    return {"base": base, "rets": rets, "extra": extra}


def metricas_estrategia(curva, r, rf_diario=0.0):
    """Métricas de riesgo/retorno de una curva de equity y su serie de retornos."""
    r = r.dropna()
    n = len(r)
    years = n / 252 if n else 0
    cagr = (curva.iloc[-1] / 100) ** (1 / years) - 1 if years > 0 else 0.0
    vol = r.std() * np.sqrt(252)
    excess = r - rf_diario
    sharpe = (excess.mean() * 252) / vol if vol > 0 else 0.0
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = (excess.mean() * 252) / downside if downside > 0 else 0.0
    dd = (curva - curva.cummax()) / curva.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    var95 = np.percentile(r, 5)
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd, "calmar": calmar, "var95": var95}


# ==================================================================== PÁGINA 5
def pagina_backtest():
    st.header("5 · Backtest y validación")
    if datos is None or datos["close"].empty or datos["close"].shape[1] < 2:
        aviso_sin_datos()
        return

    if senales_sel:
        st.info(f"**Señales activas ({len(senales_sel)}) — combinación: {modo_combinacion}**\n\n"
                + "\n".join(f"- {s}" for s in senales_sel))
    else:
        st.info("**Sin señales seleccionadas:** el overlay equivale al Buy & Hold (siempre "
                "invertido). Agrégalas en la barra lateral → *Universo y señales*.")

    bt = calcular_backtest(rf_anual, costo_oper, modo_combinacion)
    base, rets, extra = bt["base"], bt["rets"], bt["extra"]

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

    c1, c2, c3 = st.columns(3)
    c1.metric("Combinación de señales", modo_combinacion.split(" ")[0] if senales_sel else "—")
    c2.metric("Tiempo invertido (overlay)", f"{extra['pct_invertido']*100:.0f} %")
    c3.metric("Nº de operaciones", extra["n_operaciones"])

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
        m = metricas_estrategia(base[nombre], r, extra["rf_diario"])
        filas.append({
            "Estrategia": nombre,
            "CAGR": f"{m['cagr']*100:.2f} %",
            "Volatilidad": f"{m['vol']*100:.2f} %",
            "Sharpe": f"{m['sharpe']:.3f}",
            "Sortino": f"{m['sortino']:.3f}",
            "Máx. Drawdown": f"{m['max_dd']*100:.2f} %",
            "Calmar": f"{m['calmar']:.3f}" if np.isfinite(m["calmar"]) else "—",
            "VaR 95% diario": f"{m['var95']*100:.2f} %",
            "SPA p-value": p_spa if nombre == "Óptimo + señales" else "—",
        })
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
    st.caption(
        "El overlay es **long-only**: sólo puede estar invertido o en efectivo, así que en "
        "mercados alcistas su meta no es ganarle el **CAGR** al Buy & Hold (eso exigiría "
        "apalancamiento o ponerse corto), sino mejorar el **riesgo ajustado** — mayor "
        "Sharpe/Sortino/Calmar y menor *drawdown*. Su ventaja se hace visible cuando el "
        "periodo incluye caídas fuertes y el efectivo rinde la tasa libre de riesgo. "
        "SPA p-value < 0.05 sugiere que el overlay supera al Buy & Hold por algo más que azar.")


# ==================================================================== PÁGINA 6
def pagina_indicadores():
    st.header("6 · Indicadores técnicos")
    if datos is None or not datos.get("ohlc"):
        aviso_sin_datos()
        return

    activos = list(datos["ohlc"].keys())
    c0, c1 = st.columns([2, 3])
    with c0:
        activo = st.selectbox("Activo", activos, key="ind_activo")
    with c1:
        paneles = st.multiselect("Paneles inferiores",
                                 ["Volumen", "RSI", "MACD", "Estocástico"],
                                 default=["RSI", "MACD"], key="ind_paneles")

    df = datos["ohlc"][activo]
    ind = S.indicadores_para_grafico(df)

    filas = 1 + len(paneles)
    if len(paneles) > 0:
        alturas = [0.46] + [0.54 / len(paneles)] * len(paneles)
    else:
        alturas = [1.0]
    titulos = ["Precio, medias y Bollinger"] + paneles
    fig = make_subplots(rows=filas, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=alturas, subplot_titles=titulos)

    # ---- Fila 1: velas + medias + bandas de Bollinger ----
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=activo, increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["bb_upper"], name="BB sup.",
        line=dict(color="#6b7280", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["bb_lower"], name="BB inf.",
        line=dict(color="#6b7280", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(107,114,128,0.10)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["sma"], name="SMA 200",
        line=dict(color="#f59e0b", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["ema_fast"], name="EMA 8",
        line=dict(color="#3b82f6", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["ema_slow"], name="EMA 21",
        line=dict(color="#a855f7", width=1)), row=1, col=1)

    # ---- Marcadores de entrada/salida del overlay seleccionado ----
    if senales_sel:
        pos = S.posicion_combinada(df, senales_sel, modo_combinacion)
        entradas = pos[(pos == 1) & (pos.shift(1) == 0)].index
        salidas = pos[(pos == 0) & (pos.shift(1) == 1)].index
        fig.add_trace(go.Scatter(x=entradas, y=df.loc[entradas, "low"] * 0.97, mode="markers",
            name="Entrada", marker=dict(symbol="triangle-up", color="#22c55e", size=11,
            line=dict(width=1, color="white"))), row=1, col=1)
        fig.add_trace(go.Scatter(x=salidas, y=df.loc[salidas, "high"] * 1.03, mode="markers",
            name="Salida", marker=dict(symbol="triangle-down", color="#ef4444", size=11,
            line=dict(width=1, color="white"))), row=1, col=1)

    # ---- Paneles inferiores ----
    r = 2
    for panel in paneles:
        if panel == "Volumen":
            fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volumen",
                marker_color="#475569", showlegend=False), row=r, col=1)
        elif panel == "RSI":
            fig.add_trace(go.Scatter(x=df.index, y=ind["rsi"], name="RSI",
                line=dict(color="#3b82f6"), showlegend=False), row=r, col=1)
            fig.add_hline(y=70, line=dict(color="#ef4444", dash="dash", width=1), row=r, col=1)
            fig.add_hline(y=30, line=dict(color="#22c55e", dash="dash", width=1), row=r, col=1)
            fig.update_yaxes(range=[0, 100], row=r, col=1)
        elif panel == "MACD":
            fig.add_trace(go.Bar(x=df.index, y=ind["macd_hist"], name="Histograma",
                marker_color="#475569", showlegend=False), row=r, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ind["macd"], name="MACD",
                line=dict(color="#3b82f6"), showlegend=False), row=r, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ind["macd_signal"], name="Señal",
                line=dict(color="#f59e0b"), showlegend=False), row=r, col=1)
        elif panel == "Estocástico":
            fig.add_trace(go.Scatter(x=df.index, y=ind["stoch_k"], name="%K",
                line=dict(color="#3b82f6"), showlegend=False), row=r, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ind["stoch_d"], name="%D",
                line=dict(color="#f59e0b"), showlegend=False), row=r, col=1)
            fig.add_hline(y=80, line=dict(color="#ef4444", dash="dash", width=1), row=r, col=1)
            fig.add_hline(y=20, line=dict(color="#22c55e", dash="dash", width=1), row=r, col=1)
            fig.update_yaxes(range=[0, 100], row=r, col=1)
        r += 1

    fig.update_layout(height=380 + 200 * len(paneles), template=PLOTLY_TEMPLATE,
                      paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False,
                      hovermode="x unified", margin=dict(t=50, b=20, l=10, r=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

    if senales_sel:
        pos_final = int(S.posicion_combinada(df, senales_sel, modo_combinacion).iloc[-1])
        estado = "🟢 Invertido" if pos_final == 1 else "🔴 En efectivo"
        st.caption(f"Estado actual de **{activo}** según las señales activas "
                   f"({modo_combinacion}): {estado}. Los triángulos marcan las entradas y "
                   f"salidas que generaría ese overlay.")
    else:
        st.caption("Selecciona señales en la barra lateral para ver las entradas y salidas "
                   "marcadas sobre el precio.")


# ====================================================================== ROUTER
st.title("Administración de Portafolios con Señales Cuantitativas")

if pagina.startswith("1"):
    pagina_perfil()
elif pagina.startswith("2"):
    pagina_universo()
elif pagina.startswith("3"):
    pagina_frontera()
elif pagina.startswith("4"):
    pagina_asignacion()
elif pagina.startswith("5"):
    pagina_backtest()
else:
    pagina_indicadores()
