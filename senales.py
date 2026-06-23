"""
================================================================================
   LIBRERÍA DE SEÑALES CUANTITATIVAS SELECCIONABLES
================================================================================
Proyecto final - Finanzas Computacionales

Cada señal sigue el MISMO contrato de la librería modular del profesor
(09DashBoardTradingBot/estrategias): una función `ejecutar(df, ...)` que recibe
un DataFrame con columnas ['open','high','low','close','volume'] y devuelve dos
Series binarias alineadas al índice de df:

    buy_sig  : 1 = señal de compra/entrada, 0 = nada
    sell_sig : 1 = señal de venta/salida,  0 = nada

La función `senal_a_posicion(buy, sell)` convierte esas señales discretas en una
serie continua de POSICIÓN (1 = invertido, 0 = en efectivo), que es lo que el
dashboard usa como "overlay táctico" sobre los pesos del portafolio de Markowitz.

El diccionario CATALOGO_SENALES es la fuente única de verdad para poblar los
checkboxes del dashboard: nombre visible -> (callable, descripción).
================================================================================
"""

import numpy as np
import pandas as pd
import ta


# ==============================================================================
# REGLAS CERRADAS DE ANÁLISIS TÉCNICO
# ==============================================================================

def senal_rsi(df, window=14, limit_buy=30, limit_sell=70):
    """RSI: compra en sobreventa (<limit_buy), vende en sobrecompra (>limit_sell)."""
    rsi = ta.momentum.rsi(df['close'], window=window)
    buy_sig = (rsi < limit_buy).astype(int)
    sell_sig = (rsi > limit_sell).astype(int)
    return buy_sig.fillna(0).astype(int), sell_sig.fillna(0).astype(int)


def senal_macd(df, fast=12, slow=26, signal=9):
    """MACD: compra cuando MACD cruza por encima de su señal; vende en el cruce inverso."""
    macd_ind = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
    macd_line = macd_ind.macd()
    signal_line = macd_ind.macd_signal()
    buy_sig = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
    sell_sig = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
    return buy_sig.fillna(0).astype(int), sell_sig.fillna(0).astype(int)


def senal_ema(df, fast=8, slow=21):
    """Cruce de EMAs: compra cuando la EMA rápida cruza por encima de la lenta."""
    ema_fast = ta.trend.ema_indicator(df['close'], window=fast)
    ema_slow = ta.trend.ema_indicator(df['close'], window=slow)
    buy_sig = ((ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))).astype(int)
    sell_sig = ((ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))).astype(int)
    return buy_sig.fillna(0).astype(int), sell_sig.fillna(0).astype(int)


def senal_bollinger(df, window=20, dev=2.0):
    """Bandas de Bollinger: compra bajo la banda inferior, vende sobre la superior."""
    bb = ta.volatility.BollingerBands(df['close'], window=window, window_dev=dev)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    buy_sig = (df['close'] < lower).astype(int)
    sell_sig = (df['close'] > upper).astype(int)
    return buy_sig.fillna(0).astype(int), sell_sig.fillna(0).astype(int)


def senal_stoch_momentum(df, stoch_window=14, stoch_smooth=3,
                         stoch_limit_buy=20, stoch_limit_sell=80, momentum_window=14):
    """Estocástico + Momentum: compra si %K<20 y momentum>0; vende si %K>80 y momentum<0."""
    stoch = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'],
        window=stoch_window, smooth_window=stoch_smooth
    )
    stoch_k = stoch.stoch()
    momentum = df['close'].diff(momentum_window)
    buy_sig = ((stoch_k < stoch_limit_buy) & (momentum > 0)).astype(int)
    sell_sig = ((stoch_k > stoch_limit_sell) & (momentum < 0)).astype(int)
    return buy_sig.fillna(0).astype(int), sell_sig.fillna(0).astype(int)


def senal_tendencia(df, window=200):
    """Filtro de tendencia: compra cuando el precio cruza por encima de la SMA larga;
    vende cuando cruza por debajo. Mantiene al portafolio fuera de mercados bajistas."""
    sma = ta.trend.sma_indicator(df['close'], window=window)
    buy_sig = ((df['close'] > sma) & (df['close'].shift(1) <= sma.shift(1))).astype(int)
    sell_sig = ((df['close'] < sma) & (df['close'].shift(1) >= sma.shift(1))).astype(int)
    return buy_sig.fillna(0).astype(int), sell_sig.fillna(0).astype(int)


# ==============================================================================
# SEÑAL ECONOMÉTRICA / MACHINE LEARNING (opcional)
# ==============================================================================

def senal_logit(df):
    """
    Regresión logística sobre indicadores técnicos rezagados (t-1) para predecir
    la dirección del precio. Entrena con el 70% inicial y predice el 30% final
    (evita look-ahead). Devuelve compra al predecir alza y venta al predecir baja.

    Importa sklearn de forma perezosa para no exigir la dependencia si no se usa.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    f = df.copy()
    f['RSI'] = ta.momentum.rsi(f['close'], window=14)
    f['Momentum'] = f['close'].diff(14)
    macd_ind = ta.trend.MACD(f['close'])
    f['MACD'] = macd_ind.macd()
    f['Signal'] = macd_ind.macd_signal()
    f['EMA20'] = ta.trend.ema_indicator(f['close'], window=20)
    f['EMA50'] = ta.trend.ema_indicator(f['close'], window=50)
    bb = ta.volatility.BollingerBands(f['close'], window=20)
    f['BB_upper'] = bb.bollinger_hband()
    f['BB_lower'] = bb.bollinger_lband()

    feats = ['RSI', 'Momentum', 'MACD', 'Signal', 'EMA20', 'EMA50', 'BB_upper', 'BB_lower']
    lagged = []
    for c in feats:
        f[c + '_lag1'] = f[c].shift(1)
        lagged.append(c + '_lag1')

    f['Direccion'] = (f['close'] > f['close'].shift(1)).astype(int)
    clean = f[['Direccion'] + lagged].dropna()

    pred_series = pd.Series(0, index=df.index)
    if len(clean) < 40:
        return pred_series, pred_series  # datos insuficientes -> sin señales

    split = int(len(clean) * 0.7)
    X_train, y_train = clean.iloc[:split][lagged], clean.iloc[:split]['Direccion']
    X_test = clean.iloc[split:][lagged]
    test_idx = clean.iloc[split:].index

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    pred_series.loc[test_idx] = preds
    buy_sig = ((pred_series == 1) & (pred_series.shift(1) != 1)).astype(int)
    sell_sig = ((pred_series == 0) & (pred_series.shift(1) == 1)).astype(int)
    return buy_sig.fillna(0).astype(int), sell_sig.fillna(0).astype(int)


# ==============================================================================
# CATÁLOGO: fuente única de verdad para los checkboxes del dashboard
# ==============================================================================

CATALOGO_SENALES = {
    "RSI (sobreventa/sobrecompra)":      (senal_rsi,            "Compra cuando el RSI < 30 y vende cuando el RSI > 70."),
    "MACD (cruce)":                      (senal_macd,           "Compra/vende en los cruces de la línea MACD con su señal."),
    "Cruce de EMAs (8/21)":              (senal_ema,            "Compra cuando la EMA rápida cruza por encima de la lenta."),
    "Bandas de Bollinger":               (senal_bollinger,      "Compra bajo la banda inferior, vende sobre la superior."),
    "Estocástico + Momentum":            (senal_stoch_momentum, "Combina sobreventa del %K con momentum positivo."),
    "Filtro de tendencia (SMA 200)":     (senal_tendencia,      "Sólo permanece invertido cuando el precio > SMA 200."),
    "Regresión logística (ML)":          (senal_logit,          "Modelo Logit que predice la dirección con indicadores rezagados."),
}


# ==============================================================================
# CONVERSIÓN DE SEÑALES A POSICIÓN CONTINUA (overlay táctico)
# ==============================================================================

def senal_a_posicion(buy_sig, sell_sig):
    """
    Convierte señales discretas de compra/venta en una serie de posición continua:
    1 = invertido, 0 = en efectivo. Estado inicial = fuera de mercado (0).
    Al recibir una compra pasa a 1, al recibir una venta pasa a 0, y mantiene el
    estado entre señales (forward-fill del estado).
    """
    pos = np.zeros(len(buy_sig), dtype=int)
    actual = 0
    buy_vals = buy_sig.values
    sell_vals = sell_sig.values
    for i in range(len(buy_sig)):
        if buy_vals[i] == 1:
            actual = 1
        elif sell_vals[i] == 1:
            actual = 0
        pos[i] = actual
    return pd.Series(pos, index=buy_sig.index)


def posicion_combinada(df, nombres_senales):
    """
    Aplica una o varias señales seleccionadas a un activo y combina sus posiciones.
    Con varias señales se exige consenso (AND): el activo está invertido sólo si
    TODAS las señales activas coinciden en estar dentro del mercado. Si no se
    selecciona ninguna señal, devuelve posición constante = 1 (siempre invertido,
    equivale a Buy & Hold del portafolio óptimo).
    """
    if not nombres_senales:
        return pd.Series(1, index=df.index)

    posiciones = []
    for nombre in nombres_senales:
        func, _ = CATALOGO_SENALES[nombre]
        buy_sig, sell_sig = func(df)
        posiciones.append(senal_a_posicion(buy_sig, sell_sig))

    combinada = posiciones[0]
    for p in posiciones[1:]:
        combinada = combinada * p  # AND lógico (consenso)
    return combinada
