# Seleccion de portafolios y administracion de inversiones

## ¿Qué hace?

Navegación por la barra lateral (5 secciones):

| Sección | Contenido |
|---|---|
| **1 · Perfil del inversor** | Cuestionario de aversión al riesgo → perfil Conservador / Moderado / Agresivo + coeficiente λ (con gauge). |
| **2 · Universo y señales** | Descarga de precios (yfinance), métricas por activo, correlación, y selección de señales. |
| **3 · Frontera eficiente** | Markowitz con `scipy.optimize`: nube de portafolios, **acciones individuales**, mínima varianza, máximo Sharpe y el óptimo del perfil. |
| **4 · Asignación de activos** | Pesos recomendados (pie), montos en MXN y estado táctico actual de cada activo. |
| **5 · Backtest y validación** | Óptimo Buy & Hold vs. Óptimo + overlay de señales vs. equiponderado. CAGR, Sharpe, Máx. Drawdown, VaR y prueba SPA de Hansen. |

Las señales (RSI, MACD, EMA, Bollinger, Estocástico+Momentum, SMA 200, Logit) se
reutilizan de `senales.py` con el contrato `ejecutar(df) -> (buy, sell)`; con
varias activas se exige **consenso (AND)**. Este concenso funciona de forma en que las
señales seleccionadas se tienen que poner todas de acuerdo para ejecutar los movimientos


## Archivos

```
ProyectoStreamlit/
├── app.py                 # Aplicación Streamlit (las 5 secciones)
├── senales.py             # Librería de señales cuantitativas (idéntica a Proyecto/)
├── portafolio.py          # Optimización de Markowitz (idéntica a Proyecto/)
├── requirements.txt
├── .streamlit/config.toml # Tema oscuro
└── README.md
```
