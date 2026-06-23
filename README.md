# Robot de Portafolios con Señales Cuantitativas — versión Streamlit

Misma esencia que la app de Quarto/Shiny (`../Proyecto`), portada a **Streamlit**
para publicarla como **aplicación web**, con **tema oscuro**.

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
varias activas se exige **consenso (AND)**.

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

## Ejecutar localmente

```powershell
# Crear/activar un entorno e instalar dependencias
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Lanzar la app (se abre en el navegador)
streamlit run app.py
```

Uso: en la barra lateral abra **Universo y señales**, ajuste tickers y presione
**⬇️ Descargar y analizar**. Luego navegue por las secciones 3–5. El perfil y las
señales se ajustan en cualquier momento desde la barra lateral.

## Publicar como app web (Streamlit Community Cloud — gratis)

1. Suba **solo la carpeta `ProyectoStreamlit/`** a un repositorio de GitHub
   (con `app.py`, `senales.py`, `portafolio.py`, `requirements.txt` y
   `.streamlit/config.toml`).
2. En [share.streamlit.io](https://share.streamlit.io) → **New app**, elija el
   repositorio y como *Main file path* indique `app.py`.
3. Streamlit instala `requirements.txt` y publica la app. Copie la URL pública
   y entréguela junto con los archivos.

> El tema oscuro ya va en `.streamlit/config.toml`, así que se aplica tanto en
> local como en la nube. Streamlit maneja UTF-8 correctamente: **no hay** el
> problema de acentos (mojibake) que tenía la versión de Shiny en Windows.
