## librerias

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DIAS_ANIO = 252


# ==============================================================================
# ESTADÍSTICOS DE ENTRADA
# ==============================================================================

def rendimientos_diarios(precios):
    """Rendimientos aritméticos diarios a partir de una matriz de precios (DataFrame)."""
    return precios.pct_change().dropna()


def estadisticos_anualizados(precios):
    """
    Devuelve (mu, cov):
      mu  : Series con el rendimiento esperado anualizado por activo.
      cov : DataFrame con la matriz de covarianza anualizada.
    """
    rends = rendimientos_diarios(precios)
    mu = rends.mean() * DIAS_ANIO
    cov = rends.cov() * DIAS_ANIO
    return mu, cov


def metricas_activos(precios, rf=0.0):
    """Tabla por activo: rendimiento esperado, volatilidad y Sharpe (anualizados)."""
    rends = rendimientos_diarios(precios)
    mu = rends.mean() * DIAS_ANIO
    vol = rends.std() * np.sqrt(DIAS_ANIO)
    sharpe = (mu - rf) / vol
    tabla = pd.DataFrame({
        "Activo": mu.index,
        "Rend. esperado": mu.values,
        "Volatilidad": vol.values,
        "Sharpe": sharpe.values,
    }).reset_index(drop=True)
    return tabla


# ==============================================================================
# MÉTRICAS DE UN PORTAFOLIO
# ==============================================================================

def rendimiento_portafolio(w, mu):
    return float(np.dot(w, mu))


def volatilidad_portafolio(w, cov):
    return float(np.sqrt(np.dot(w, np.dot(cov, w))))


def sharpe_portafolio(w, mu, cov, rf=0.0):
    vol = volatilidad_portafolio(w, cov)
    if vol == 0:
        return 0.0
    return (rendimiento_portafolio(w, mu) - rf) / vol


# ==============================================================================
# FRONTERA EFICIENTE POR SIMULACIÓN
# ==============================================================================

def frontera_simulada(mu, cov, n_port=5000, rf=0.0, semilla=42):
    """
    Genera una nube de n_port portafolios aleatorios long-only.
    Devuelve un DataFrame con columnas: rendimiento, volatilidad, sharpe y los
    pesos de cada activo (una columna w_<activo>).
    """
    rng = np.random.default_rng(semilla)
    n = len(mu)
    activos = list(mu.index)
    mu_v = mu.values
    cov_v = cov.values

    filas = []
    for _ in range(n_port):
        w = rng.random(n)
        w = w / w.sum()
        r = float(np.dot(w, mu_v))
        v = float(np.sqrt(np.dot(w, np.dot(cov_v, w))))
        s = (r - rf) / v if v > 0 else 0.0
        fila = {"rendimiento": r, "volatilidad": v, "sharpe": s}
        for i, a in enumerate(activos):
            fila[f"w_{a}"] = w[i]
        filas.append(fila)
    return pd.DataFrame(filas)


# ==============================================================================
# OPTIMIZACIÓN CON SCIPY
# ==============================================================================

def _restricciones_base(n):
    """Restricción de suma de pesos = 1 y cotas long-only [0, 1]."""
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.repeat(1.0 / n, n)
    return cons, bounds, w0


def optimo_min_varianza(mu, cov):
    """Portafolio de mínima varianza global."""
    n = len(mu)
    cov_v = cov.values
    cons, bounds, w0 = _restricciones_base(n)
    obj = lambda w: np.dot(w, np.dot(cov_v, w))
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=mu.index)


def optimo_max_sharpe(mu, cov, rf=0.0):
    """Portafolio tangente (máxima razón de Sharpe)."""
    n = len(mu)
    mu_v = mu.values
    cov_v = cov.values
    cons, bounds, w0 = _restricciones_base(n)

    def neg_sharpe(w):
        r = np.dot(w, mu_v)
        v = np.sqrt(np.dot(w, np.dot(cov_v, w)))
        return -(r - rf) / v if v > 0 else 0.0

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=mu.index)


def optimo_por_aversion(mu, cov, lam):
    """
    Maximiza la utilidad media-varianza  U(w) = w'mu - 0.5 * lam * w'cov w.
    'lam' es el coeficiente de aversión al riesgo que entrega el perfilamiento:
    lam bajo  -> agresivo (busca rendimiento), lam alto -> conservador (penaliza riesgo).
    """
    n = len(mu)
    mu_v = mu.values
    cov_v = cov.values
    cons, bounds, w0 = _restricciones_base(n)
    obj = lambda w: -(np.dot(w, mu_v) - 0.5 * lam * np.dot(w, np.dot(cov_v, w)))
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=mu.index)


def resumen_portafolio(nombre, w, mu, cov, rf=0.0):
    """Fila resumen con rendimiento, volatilidad y Sharpe de un portafolio dado."""
    return {
        "Portafolio": nombre,
        "Rend. esperado": rendimiento_portafolio(w.values, mu.values),
        "Volatilidad": volatilidad_portafolio(w.values, cov.values),
        "Sharpe": sharpe_portafolio(w.values, mu.values, cov.values, rf),
    }
