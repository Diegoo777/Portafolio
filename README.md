# Robot de Administración de Portafolios con Señales Cuantitativas

Aplicación web de **algorithmic trading** que combina la **Teoría Moderna de Portafolios
(Markowitz)** con un *overlay* táctico de **señales cuantitativas** de análisis técnico.
Proyecto escolar de Finanzas Computacionales / Algorithmic Trading, construido con
**Streamlit** y pensado para desplegarse en **Streamlit Community Cloud**.

El flujo completo: perfilas al inversor → eliges un universo de activos → optimizas el
portafolio en la frontera eficiente → asignas el capital → validas la estrategia con un
*backtest* → y exploras los indicadores técnicos de cada activo.

> ⚠️ **Aviso:** es un trabajo académico con fines educativos. **No es asesoría financiera**
> ni una recomendación de inversión.

---

## 🚀 App en línea

Una vez desplegada en Streamlit Community Cloud, la app vive en una URL pública del tipo
`https://<tu-app>.streamlit.app`. Coloca aquí el enlace cuando la publiques:

[![Abrir en Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

---

## ¿Qué hace?

Navegación por la barra lateral (6 secciones):

| Sección | Contenido |
|---|---|
| **1 · Perfil del inversor** | Cuestionario de aversión al riesgo → perfil Conservador / Moderado / Agresivo + coeficiente de aversión λ (con *gauge*). |
| **2 · Universo y señales** | Descarga de precios (yfinance), métricas anualizadas por activo, matriz de correlación y selección de señales + modo de combinación. |
| **3 · Frontera eficiente** | Markowitz con `scipy.optimize`: nube de portafolios (esquema *spring*), acciones individuales, mínima varianza, máximo Sharpe y el óptimo del perfil. |
| **4 · Asignación de activos** | Pesos recomendados (*pie*), montos en MXN y estado táctico actual (Invertido / Efectivo) de cada activo. |
| **5 · Backtest y validación** | Óptimo Buy & Hold vs. Óptimo + señales vs. equiponderado. Efectivo que rinde la tasa libre de riesgo, costos de transacción, CAGR, Sharpe, Sortino, Calmar, Máx. Drawdown, VaR y prueba **SPA de Hansen**. |
| **6 · Indicadores técnicos** | Velas (*candlestick*) por activo con SMA 200, EMA 8/21 y Bandas de Bollinger, marcadores de entrada/salida y paneles de RSI, MACD, Estocástico y volumen. |

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
  rezagados (requiere `scikit-learn`; si no está instalado, la señal se desactiva sola).

**Combinación de varias señales** (configurable en la barra lateral):

- **Consenso (AND)** — invertido sólo si **todas** coinciden (más tiempo en efectivo).
- **Cualquiera (OR)** — invertido si **cualquiera** lo indica.
- **Mayoría (voto)** — invertido si al menos la mitad coincide.

> 💡 Una estrategia *long-only* sólo puede estar **invertida o en efectivo**: en mercados
> alcistas su meta no es ganarle el CAGR al Buy & Hold, sino mejorar el **riesgo ajustado**
> (mayor Sharpe/Sortino/Calmar y menor *drawdown*). Su ventaja aparece cuando el periodo
> incluye caídas fuertes y el efectivo rinde la tasa libre de riesgo.

---

## Arquitectura

Tres módulos con la matemática desacoplada de la interfaz:

- **`app.py`** — única parte que depende de Streamlit. Interfaz, caché y estado de las 6 secciones.
- **`portafolio.py`** — módulo puro de Markowitz (numpy/pandas/scipy). Portafolios *long-only*
  (pesos ≥ 0 que suman 1), estadísticos anualizados (252 días), frontera eficiente simulada y
  tres óptimos por `scipy.optimize` (mínima varianza, máximo Sharpe, utilidad media-varianza por λ).
- **`senales.py`** — librería pura de señales técnicas (usa `ta`; `scikit-learn` opcional).

---

## ▶️ Ejecutar localmente

Requisitos: Python 3.11.

```bash
# 1. Clonar el repositorio
git clone https://github.com/Diegoo777/Portafolio.git
cd Portafolio

# 2. (Opcional) crear y activar un entorno
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS / Linux

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
streamlit run app.py
```

La app abre en `http://localhost:8501`. Con **conda**:

```bash
conda create -n trading python=3.11 -y
conda activate trading
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Desplegar en Streamlit Community Cloud

Para convertir el repositorio en una página web pública y gratuita:

1. **Sube el proyecto a GitHub** (este repositorio ya lo está).
2. Entra a **[share.streamlit.io](https://share.streamlit.io)** e inicia sesión con tu cuenta de GitHub.
3. Pulsa **"Create app" → "Deploy a public app from GitHub"**.
4. Configura:
   - **Repository:** `Diegoo777/Portafolio`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. (Opcional) En **Advanced settings** elige **Python 3.11** para igualar el entorno local.
6. Pulsa **Deploy**. Streamlit instala automáticamente lo de `requirements.txt` y publica la app.

No se necesitan *secrets* ni claves de API: los precios se descargan en vivo desde Yahoo
Finance con `yfinance`. El tema oscuro se toma de `.streamlit/config.toml`.

---

## 📁 Estructura del proyecto

```
Portafolio/
├── app.py                  # Aplicación Streamlit (las 6 secciones)
├── senales.py              # Librería de señales cuantitativas + indicadores
├── portafolio.py           # Optimización de Markowitz
├── requirements.txt        # Dependencias
├── .streamlit/
│   └── config.toml         # Tema oscuro
├── .devcontainer/          # Configuración para GitHub Codespaces
├── CLAUDE.md               # Guía para asistentes de IA en el repo
└── README.md
```

---

## 🛠️ Tecnologías

- **[Streamlit](https://streamlit.io)** — interfaz web.
- **[yfinance](https://github.com/ranaroussi/yfinance)** — datos de mercado.
- **pandas · numpy · scipy** — datos y optimización.
- **[plotly](https://plotly.com/python/)** — gráficas interactivas.
- **[ta](https://github.com/bukosabino/ta)** — indicadores de análisis técnico.
- **scikit-learn** — señal de regresión logística (opcional).
- **[arch](https://github.com/bashtage/arch)** — prueba SPA de Hansen (opcional).
