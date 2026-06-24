# Optimizador de portafolios y analisis tecnico

### Participantes
Abigail Sampedro Gutiérrez

Diego Pedraza Barajas

Aplicación web de **algorithmic trading** que aplica teoria de portafolios interactivo usando
**señales cuantitativas** de análisis técnico.
Proyecto final de Finanzas computacionales y algorithmic trading utilizando streamlit

comportamiento de la app: perfilas al inversor → eliges un universo de activos → optimizas el
portafolio en la frontera eficiente → asignas el capital → validas la estrategia con un
*backtest* → y exploras los indicadores técnicos de cada activo.


---

##  App en línea

El link para acceder a la aplicacion es el siguiente:

[App web](https://portafoliosalgorithmictrading.streamlit.app)

---

## ¿Qué hace?

Navegación por la barra lateral (6 secciones):

| Sección | Contenido |
|---|---|
| **1 · Perfil del inversor** | Cuestionario de aversión al riesgo → perfil Conservador / Moderado / Agresivo + coeficiente de aversión λ. |
| **2 · Universo y señales** | Descarga de precios (yfinance), métricas anualizadas por activo, matriz de correlación y selección de señales + modo de combinación. |
| **3 · Frontera eficiente** | Markowitz: nube de portafolios, acciones individuales, mínima varianza, máximo Sharpe y el óptimo del perfil. |
| **4 · Asignación de activos** | Pesos recomendados, montos en MXN y estado actual (Invertido(compra)  o Efectivo(venta)) de cada activo. |
| **5 · Backtest y validación** | Óptimo Buy & Hold vs. Óptimo + señales vs. equiponderado. Efectivo que rinde la tasa libre de riesgo, costos de transacción, CAGR, Sharpe, Sortino, Calmar, Máx. Drawdown, VaR y pruebaSPA de Hansen. |
| **6 · Indicadores técnicos** | Velas japonesas por activo con SMA 200, EMA 8/21 y Bandas de Bollinger, marcadores de entrada/salida y paneles de RSI, MACD, Estocástico y volumen. |

---

## Señales disponibles

Catálogo en `senales.py` (cada una sigue el contrato `ejecutar(df) -> (buy, sell)`):

- **RSI** — sobreventa / sobrecompra (compra <30, vende >70).
- **MACD** — cruces de la línea con su señal.
- **Cruce de EMAs (8/21)** — la EMA rápida cruza a la lenta.
- **Bandas de Bollinger** — reversión a la media en los extremos.
- **Estocástico + Momentum** — sobreventa del %K con momentum positivo.
- **Filtro de tendencia (SMA 200)** — *filtro de régimen* estilo Faber: invertido sólo
  mientras el precio está por encima de su media de 200 días (esquiva mercados bajistas).
- **Regresión logística (ML)** — modelo Logit que predice la dirección con indicadores
  rezagados.

**Combinación de varias señales** (configurable en la barra lateral):

- **Consenso (AND)** — invertido sólo si **todas** coinciden (más tiempo en efectivo).
- **Cualquiera (OR)** — invertido si **cualquiera** lo indica.
- **Mayoría (voto)** — invertido si al menos la mitad coincide.


---

## Arquitectura


- **`app.py`** — única parte que depende de Streamlit. Interfaz, caché y estado de las 6 secciones.
- **`portafolio.py`** — módulo puro de Markowitz (numpy/pandas/scipy). Portafolios *long-only*
  (pesos ≥ 0 que suman 1), estadísticos anualizados (252 días), frontera eficiente simulada y
  tres óptimos por scipy.optimize (mínima varianza, máximo Sharpe, utilidad media-varianza por λ).
- **`senales.py`** — librería pura de señales técnicas (usa `ta`; `scikit-learn`).

---


##  Estructura del proyecto

```
Portafolio/
├── app.py                  # Aplicación Streamlit (las 6 secciones)
├── senales.py              # Librería de señales cuantitativas + indicadores
├── portafolio.py           # Optimización de Markowitz
├── requirements.txt        # Dependencias para streamlit
├── .streamlit/
│   └── config.toml         # configuracion de streamlit
└── README.md
```


