#Librerías

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import ta


#Reglas de análisis técnico (cerradas)

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


def posicion_tendencia(df, window=200):
    """
    Posición de un Filtro de regimén: invertido (1) mientras el precio está por
    encima de su SMA larga, en efectivo (0) cuando está por debajo. Es la regla
    de 'time-series momentum' / media móvil de 200 días (estilo Faber): su valor
    NO es ganarle en rendimiento al Buy & Hold en mercados alcistas, sino esquivar
    los grandes descensos.

    El periodo de calentamiento (SMA aún sin definir) se asume INVERTIDO, de modo
    que el overlay arranca igual que el Buy & Hold y no arrastra efectivo ocioso.
    """
    sma = ta.trend.sma_indicator(df['close'], window=window)
    invertido = (df['close'] > sma)
    invertido[sma.isna()] = True  # calentamiento -> invertido (equivale a B&H)
    return invertido.astype(int)


def senal_tendencia(df, window=200):
    """Eventos compra/venta del filtro de tendencia: son las transiciones del
    estado de `posicion_tendencia` (cruce al alza = compra, cruce a la baja = venta).
    Se usan sólo para dibujar marcadores; la exposición real la da `posicion_tendencia`."""
    pos = posicion_tendencia(df, window=window)
    buy_sig = ((pos == 1) & (pos.shift(1) == 0)).fillna(0).astype(int)
    sell_sig = ((pos == 0) & (pos.shift(1) == 1)).fillna(0).astype(int)
    return buy_sig, sell_sig


#Señal econométrica

def senal_logit(df):
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        vacio = pd.Series(0, index=df.index)
        return vacio, vacio

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


#Catálogo de señales

@dataclass
class Senal:
    """Metadatos de una señal seleccionable.

    func           : función `ejecutar(df) -> (buy, sell)` (para marcadores).
    desc           : descripción visible en la interfaz.
    tipo           : 'tendencia' | 'reversion' | 'cruce' | 'ml' (informativo).
    estado_inicial : 1 = el overlay arranca INVERTIDO (asume que ya se posee el
                     portafolio óptimo y la señal sólo dice cuándo salir/entrar);
                     evita el arrastre de efectivo del arranque en 0.
    posicion       : función opcional `posicion(df) -> Series 0/1` que devuelve la
                     exposición directa (útil para filtros de régimen). Si es None
                     se deriva de (buy, sell) con `senal_a_posicion`.
    """
    func: Callable
    desc: str
    tipo: str = "evento"
    estado_inicial: int = 1
    posicion: Optional[Callable] = None


CATALOGO_SENALES = {
    "RSI (sobreventa/sobrecompra)":  Senal(senal_rsi,            "Compra cuando el RSI < 30 y vende cuando el RSI > 70.", tipo="reversion"),
    "MACD (cruce)":                  Senal(senal_macd,           "Compra/vende en los cruces de la línea MACD con su señal.", tipo="cruce"),
    "Cruce de EMAs (8/21)":          Senal(senal_ema,            "Compra cuando la EMA rápida cruza por encima de la lenta.", tipo="cruce"),
    "Bandas de Bollinger":           Senal(senal_bollinger,      "Compra bajo la banda inferior, vende sobre la superior.", tipo="reversion"),
    "Estocástico + Momentum":        Senal(senal_stoch_momentum, "Combina sobreventa del %K con momentum positivo.", tipo="reversion"),
    "Filtro de tendencia (SMA 200)": Senal(senal_tendencia,      "Sólo permanece invertido cuando el precio > SMA 200.", tipo="tendencia", posicion=posicion_tendencia),
    "Regresión logística (ML)":      Senal(senal_logit,          "Modelo Logit que predice la dirección con indicadores rezagados.", tipo="ml"),
}


#Conversión de señales a posición única

def senal_a_posicion(buy_sig, sell_sig, estado_inicial=0):
    """
    Convierte señales discretas de compra/venta en una serie de posición continua:
    1 = invertido, 0 = en efectivo. `estado_inicial` fija el estado antes de la
    primera señal (1 = ya invertido, recomendado para un overlay).
    Al recibir una compra pasa a 1, al recibir una venta pasa a 0, y mantiene el
    estado entre señales (forward-fill del estado).
    """
    pos = np.empty(len(buy_sig), dtype=int)
    actual = int(estado_inicial)
    buy_vals = buy_sig.values
    sell_vals = sell_sig.values
    for i in range(len(buy_sig)):
        if buy_vals[i] == 1:
            actual = 1
        elif sell_vals[i] == 1:
            actual = 0
        pos[i] = actual
    return pd.Series(pos, index=buy_sig.index)


def posicion_de_senal(df, nombre):
    """Exposición 0/1 de UNA señal. Usa la posición directa si la señal la define
    (filtros de régimen); si no, la deriva de sus eventos buy/sell respetando el
    `estado_inicial` (por defecto, arranca invertido)."""
    s = CATALOGO_SENALES[nombre]
    if s.posicion is not None:
        return s.posicion(df).reindex(df.index).ffill().fillna(1).astype(int)
    buy_sig, sell_sig = s.func(df)
    return senal_a_posicion(buy_sig, sell_sig, estado_inicial=s.estado_inicial)


def posicion_combinada(df, nombres_senales, modo="Consenso (AND)"):
    """
    Combina la exposición de una o varias señales sobre un activo. `modo` decide
    cómo se concilian cuando hay varias:
      - "Consenso (AND)" : invertido sólo si TODAS coinciden (más tiempo en efectivo).
      - "Cualquiera (OR)": invertido si CUALQUIERA lo indica (más tiempo invertido).
      - "Mayoría (voto)" : invertido si al menos la mitad de las señales coincide.
    Sin señales seleccionadas devuelve posición constante = 1 (siempre invertido,
    equivale al Buy & Hold del portafolio óptimo).
    """
    if not nombres_senales:
        return pd.Series(1, index=df.index)

    posiciones = [posicion_de_senal(df, n) for n in nombres_senales]
    M = pd.concat(posiciones, axis=1).ffill().fillna(1)

    if modo.startswith("Cualquiera"):
        combinada = M.max(axis=1)
    elif modo.startswith("May"):
        combinada = (M.mean(axis=1) >= 0.5).astype(int)
    else:  # Consenso (AND)
        combinada = M.min(axis=1)
    return combinada.astype(int)


#Indicadores gráficos

def indicadores_para_grafico(df, sma_window=200, ema_fast=8, ema_slow=21,
                             bb_window=20, bb_dev=2.0, rsi_window=14,
                             stoch_window=14, stoch_smooth=3):
    """Calcula de una sola pasada todos los indicadores que dibuja la pestaña de
    gráficas técnicas y los devuelve en un dict de Series alineadas a `df.index`."""
    close, high, low = df['close'], df['high'], df['low']
    bb = ta.volatility.BollingerBands(close, window=bb_window, window_dev=bb_dev)
    macd = ta.trend.MACD(close)
    stoch = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=stoch_window, smooth_window=stoch_smooth)
    return {
        "sma": ta.trend.sma_indicator(close, window=sma_window),
        "ema_fast": ta.trend.ema_indicator(close, window=ema_fast),
        "ema_slow": ta.trend.ema_indicator(close, window=ema_slow),
        "bb_upper": bb.bollinger_hband(),
        "bb_lower": bb.bollinger_lband(),
        "bb_mid": bb.bollinger_mavg(),
        "rsi": ta.momentum.rsi(close, window=rsi_window),
        "macd": macd.macd(),
        "macd_signal": macd.macd_signal(),
        "macd_hist": macd.macd_diff(),
        "stoch_k": stoch.stoch(),
        "stoch_d": stoch.stoch_signal(),
    }
