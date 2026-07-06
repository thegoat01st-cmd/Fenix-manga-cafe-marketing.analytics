import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os

st.set_page_config(page_title="Fénix Manga Café — Marketing Intelligence", layout="wide", initial_sidebar_state="expanded")

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
body, .stApp { background-color: #111111; }
[data-testid="stMetric"] {
    background-color: #1c1c1c;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] { color: #888888 !important; }
[data-testid="stMetricValue"] { color: #f0f0f0 !important; }
section[data-testid="stSidebar"] {
    background-color: #1a1a1a;
}
h1, h2, h3, h4, h5, h6, p, span, label, div {
    color: #f0f0f0;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
    border: 1px solid rgba(212,175,55,0.4) !important;
    border-radius: 8px;
}
[data-testid="stTable"] table { color: #f0f0f0; }
hr { border-color: rgba(255,255,255,0.08); }
.card-highlight {
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.card-highlight h2 { margin: 0 0 6px 0; }
.card-highlight p { margin: 0; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTES
# =========================================================
COLORES_SEGMENTO = {
    "Premium": "#D4AF37",
    "Otaku Core": "#c0392b",
    "Digitales": "#2980b9",
    "Ocasionales": "#7f8c8d"
}

EMOJI_SEGMENTO = {
    "Premium": "💎",
    "Otaku Core": "🔥",
    "Ocasionales": "☕",
    "Digitales": "📱"
}

ARMS = [
    "Combo Temático Anime",
    "Invitación Evento Exclusivo",
    "Programa Fidelización",
    "Descuento Flash 24h",
    "Pack Merchandising",
    "Happy Hour Temático"
]

TS_DATA = {
    "Premium": {
        "ganador": "Invitación Evento Exclusivo",
        "conv_rate": 35.8, "reward": 146, "regret": 26.6, "pulls_ganador": 215,
        "arms": {
            "Combo Temático Anime":         {"pulls": 22,  "rewards": 2,  "alpha": 3,  "beta": 21,  "conv": 9.1},
            "Invitación Evento Exclusivo":  {"pulls": 215, "rewards": 77, "alpha": 78, "beta": 139, "conv": 35.8},
            "Programa Fidelización":        {"pulls": 116, "rewards": 32, "alpha": 33, "beta": 85,  "conv": 27.6},
            "Descuento Flash 24h":          {"pulls": 20,  "rewards": 2,  "alpha": 3,  "beta": 19,  "conv": 10.0},
            "Pack Merchandising":           {"pulls": 116, "rewards": 33, "alpha": 34, "beta": 84,  "conv": 28.4},
            "Happy Hour Temático":          {"pulls": 11,  "rewards": 0,  "alpha": 1,  "beta": 12,  "conv": 0.0},
        },
        "convergencia": {
            "rondas": [50,100,150,200,250,300,350,400,450,500],
            "reward": [12,21,37,54,67,82,95,111,128,146],
            "regret": [4.44,8.76,11.96,15.30,18.50,21.63,23.36,25.26,26.00,26.58]
        }
    },
    "Otaku Core": {
        "ganador": "Combo Temático Anime",
        "conv_rate": 35.8, "reward": 159, "regret": 42.7, "pulls_ganador": 279,
        "arms": {
            "Combo Temático Anime":         {"pulls": 279, "rewards": 100, "alpha": 101, "beta": 180, "conv": 35.8},
            "Invitación Evento Exclusivo":  {"pulls": 72,  "rewards": 24,  "alpha": 25,  "beta": 49,  "conv": 33.3},
            "Programa Fidelización":        {"pulls": 26,  "rewards": 5,   "alpha": 6,   "beta": 22,  "conv": 19.2},
            "Descuento Flash 24h":          {"pulls": 42,  "rewards": 9,   "alpha": 10,  "beta": 34,  "conv": 21.4},
            "Pack Merchandising":           {"pulls": 47,  "rewards": 14,  "alpha": 15,  "beta": 34,  "conv": 29.8},
            "Happy Hour Temático":          {"pulls": 34,  "rewards": 7,   "alpha": 8,   "beta": 28,  "conv": 20.6},
        },
        "convergencia": {
            "rondas": [50,100,150,200,250,300,350,400,450,500],
            "reward": [20,31,50,67,83,99,111,130,145,159],
            "regret": [4.43,10.47,15.54,20.28,24.11,28.22,32.45,36.57,40.13,42.74]
        }
    },
    "Ocasionales": {
        "ganador": "Happy Hour Temático",
        "conv_rate": 22.4, "reward": 95, "regret": 27.5, "pulls_ganador": 143,
        "arms": {
            "Combo Temático Anime":         {"pulls": 24,  "rewards": 3,  "alpha": 4,  "beta": 22,  "conv": 12.5},
            "Invitación Evento Exclusivo":  {"pulls": 15,  "rewards": 0,  "alpha": 1,  "beta": 16,  "conv": 0.0},
            "Programa Fidelización":        {"pulls": 40,  "rewards": 5,  "alpha": 6,  "beta": 36,  "conv": 12.5},
            "Descuento Flash 24h":          {"pulls": 253, "rewards": 54, "alpha": 55, "beta": 200, "conv": 21.3},
            "Pack Merchandising":           {"pulls": 25,  "rewards": 1,  "alpha": 2,  "beta": 25,  "conv": 4.0},
            "Happy Hour Temático":          {"pulls": 143, "rewards": 32, "alpha": 33, "beta": 112, "conv": 22.4},
        },
        "convergencia": {
            "rondas": [50,100,150,200,250,300,350,400,450,500],
            "reward": [7,16,29,40,49,56,68,77,88,95],
            "regret": [5.61,10.03,11.92,12.96,14.78,17.28,19.82,22.79,25.47,27.55]
        }
    },
    "Digitales": {
        "ganador": "Descuento Flash 24h",
        "conv_rate": 27.3, "reward": 113, "regret": 17.4, "pulls_ganador": 271,
        "arms": {
            "Combo Temático Anime":         {"pulls": 43,  "rewards": 8,  "alpha": 9,  "beta": 36,  "conv": 18.6},
            "Invitación Evento Exclusivo":  {"pulls": 49,  "rewards": 8,  "alpha": 9,  "beta": 42,  "conv": 16.3},
            "Programa Fidelización":        {"pulls": 56,  "rewards": 11, "alpha": 12, "beta": 46,  "conv": 19.6},
            "Descuento Flash 24h":          {"pulls": 271, "rewards": 74, "alpha": 75, "beta": 198, "conv": 27.3},
            "Pack Merchandising":           {"pulls": 33,  "rewards": 4,  "alpha": 5,  "beta": 30,  "conv": 12.1},
            "Happy Hour Temático":          {"pulls": 48,  "rewards": 8,  "alpha": 9,  "beta": 41,  "conv": 16.7},
        },
        "convergencia": {
            "rondas": [50,100,150,200,250,300,350,400,450,500],
            "reward": [12,25,39,46,60,72,82,94,103,113],
            "regret": [3.24,5.80,6.43,8.40,9.83,10.82,12.60,13.98,15.75,17.36]
        }
    },
}

RADAR_DIMENSIONES = ["Gasto", "Frecuencia", "Afinidad Anime", "Eventos", "Interacción Digital", "Merchandising"]
RADAR_VALORES = {
    "Premium":     [95, 90, 70, 85, 60, 88],
    "Otaku Core":  [65, 75, 98, 92, 80, 85],
    "Digitales":   [55, 40, 72, 30, 97, 45],
    "Ocasionales": [30, 20, 45, 15, 35, 20],
}

INSIGHTS = {
    "Premium": {
        "presupuesto": "S/.4,800/mes",
        "canal": "WhatsApp + Email",
        "insight": "El segmento Premium valora la exclusividad. Los eventos VIP generan 3x más conversión.",
        "kpi": "Frec 2.7x→3.5x | Valor trim S/.542→S/.700",
        "tacticas": [
            ("Evento Exclusivo Mensual", "Cena temática manga / noche anime premium", "WhatsApp+Email", "S/.2,500–4,000", "5.5x"),
            ("Pack Merch Premium", "Caja sorpresa edición limitada. Preventa para Premium", "Email+WhatsApp", "S/.800–1,500", "3.8x"),
            ("Programa Gold", "Puntos x visita. Beneficios VIP exclusivos", "CRM+App", "S/.400–600/mes", "3.2x"),
            ("Newsletter Premium", "Contenido curado: estrenos, eventos, merch", "Email", "S/.100–200/mes", "2.5x"),
        ]
    },
    "Otaku Core": {
        "presupuesto": "S/.7,500/mes",
        "canal": "WhatsApp + Instagram",
        "insight": "Otaku Core ama el contenido anime. Todos los arms tienen conversión alta +30%.",
        "kpi": "Frec 6.6x→7.5x | Valor trim S/.1,354→S/.1,600",
        "tacticas": [
            ("Combo Anime del Mes", "Bebida + snack temático + acceso a screening", "WhatsApp+IG", "S/.1,500–2,500", "5.5x"),
            ("Club Elite Fénix", "Early access eventos, voto en menú colaborativo", "App+WhatsApp", "S/.600–1,000/mes", "4.8x"),
            ("Torneos Mensuales", "Gaming, trivia anime, karaoke OP. Premios merch", "IG Stories+WA", "S/.2,000–3,500", "4.2x"),
            ("UGC Embajadores", "Programa creadores contenido. Reels, fotos. S/.10 crédito", "Instagram+TikTok", "S/.300–500", "5.0x"),
        ]
    },
    "Ocasionales": {
        "presupuesto": "S/.3,500/mes",
        "canal": "WhatsApp + Instagram",
        "insight": "Ocasionales responden a incentivos económicos. Evitar eventos complejos.",
        "kpi": "Frec 2.7x→4.0x | Satisfacción 6.0→7.5",
        "tacticas": [
            ("Happy Hour Temático", "3–6pm descuento en bebidas. Publicar en IG Stories semanal", "Instagram Stories", "S/.400–800", "3.5x"),
            ("Descuento Flash", "WhatsApp 10am: 2x1 snack del día. Válido ese día", "WhatsApp", "S/.100–200/mes", "3.2x"),
            ("Trae un Amigo", "Descuento 30% para acompañante. Efecto viral", "WhatsApp", "~S/.30/conv", "4.5x"),
            ("Encuesta + Drink", "Survey satisfacción + drink gratis. Mejora CSAT", "WhatsApp", "S/.150–300", "Indirecto"),
        ]
    },
    "Digitales": {
        "presupuesto": "S/.5,200/mes",
        "canal": "Instagram + WhatsApp",
        "insight": "Digitales convierten con urgencia y descuentos. Alto engagement como amplificadores.",
        "kpi": "Frec 3.4x→5.0x | IG Interacc 24→35",
        "tacticas": [
            ("Flash Sale Digital", "Descuento 25% por 24h en IG Stories y WhatsApp", "IG Stories+WA", "S/.500–900/mes", "3.8x"),
            ("#FenixChallenge", "Foto en local = S/.10 crédito. Viral orgánico", "Instagram+TikTok", "S/.200–400", "5.0x"),
            ("Puntos por Check-in", "5 check-ins IG = producto gratis. Frecuencia sube", "App+IG", "S/.300–500/mes", "3.5x"),
            ("Notif. Push Exclusivas", "Primeros en enterarse de eventos y ofertas", "App Push+WA", "S/.100–200/mes", "2.8x"),
        ]
    },
}

CLUSTER_TO_SEGMENTO = {1: "Otaku Core", 3: "Premium", 2: "Digitales", 0: "Ocasionales"}

FEATURES = [
    "frecuencia_visita_mes", "gasto_promedio_visita", "nivel_fan_anime",
    "interacciones_instagram_mes", "valor_cliente_trim",
    "porcentaje_compras_promocion", "intencion_recompra", "satisfaccion_general"
]

EXCEL_PATH_CANDIDATES = [
    "Base_de_datos__-_Fénix_Manga_Café.xlsx",
    "Base_de_datos - Fénix Manga Café.xlsx",
]


def _normalizar(texto):
    import unicodedata
    txt = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in txt if not unicodedata.combining(c)).lower()


def buscar_excel():
    """Busca el archivo Excel en el directorio del script y en el directorio actual,
    tolerando diferencias de tildes, espacios o guiones en el nombre."""
    dirs_a_buscar = []
    try:
        dirs_a_buscar.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    dirs_a_buscar.append(os.getcwd())

    vistos = set()
    for d in dirs_a_buscar:
        if d in vistos or not os.path.isdir(d):
            continue
        vistos.add(d)
        for nombre in os.listdir(d):
            if not nombre.lower().endswith(".xlsx"):
                continue
            norm = _normalizar(nombre)
            if "base_de_datos" in norm and "manga" in norm:
                return os.path.join(d, nombre)
            if "fenix" in norm and "manga" in norm and "cafe" in norm:
                return os.path.join(d, nombre)
    return None

# =========================================================
# ESTILO MATPLOTLIB
# =========================================================
def estilizar_ejes(ax):
    ax.set_facecolor('#1c1c1c')
    ax.tick_params(colors='#888888')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.title.set_color('#f0f0f0')
    ax.xaxis.label.set_color('#888888')
    ax.yaxis.label.set_color('#888888')

# =========================================================
# CARGA Y PROCESAMIENTO DE DATOS
# =========================================================
@st.cache_data
def cargar_datos(archivo_subido=None):
    if archivo_subido is not None:
        df = pd.read_excel(archivo_subido, sheet_name="fenix_manga_cafe_10k")
        df = df.dropna(subset=FEATURES).copy()
        X = df[FEATURES].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = km.fit_predict(X_scaled)
        df["cluster"] = clusters
        df["segmento"] = df["cluster"].map(CLUSTER_TO_SEGMENTO)
        return df

    path = None
    for p in EXCEL_PATH_CANDIDATES:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        path = buscar_excel()
    if path is None:
        return None

    df = pd.read_excel(path, sheet_name="fenix_manga_cafe_10k")
    df = df.dropna(subset=FEATURES).copy()

    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = km.fit_predict(X_scaled)
    df["cluster"] = clusters
    df["segmento"] = df["cluster"].map(CLUSTER_TO_SEGMENTO)

    return df


df = cargar_datos()

if df is None:
    st.error(
        "⚠️ No se encontró el archivo **Base_de_datos__-_Fénix_Manga_Café.xlsx** "
        "en la carpeta de `app.py`.\n\n"
        "Puedes subirlo manualmente aquí abajo, o verificar que el archivo Excel "
        "esté en el mismo directorio que `app.py` y que su nombre contenga "
        "'Fenix', 'Manga' y 'Cafe' (con o sin tildes)."
    )
    archivo_subido = st.file_uploader(
        "Sube el archivo Excel de la base de datos (.xlsx)", type=["xlsx"]
    )
    if archivo_subido is not None:
        df = cargar_datos(archivo_subido=archivo_subido)

    if df is None:
        st.stop()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🍥 Fénix Manga Café")
st.sidebar.caption("Marketing Intelligence Platform")
st.sidebar.divider()

seccion = st.sidebar.radio(
    "Navegación",
    [
        "📊 Executive Dashboard",
        "👥 Segmentación de Clientes",
        "🎯 Motor Thompson Sampling",
        "📈 Performance del Modelo",
        "💡 Insights y Recomendaciones",
    ]
)

# =========================================================
# 1. EXECUTIVE DASHBOARD
# =========================================================
if seccion == "📊 Executive Dashboard":
    st.markdown("## 📊 Executive Dashboard")
    st.caption("Visión general del negocio y resultados del modelo de optimización")

    reward_total = sum(v["reward"] for v in TS_DATA.values())
    mejor_global = max(TS_DATA.items(), key=lambda kv: kv[1]["conv_rate"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Clientes", f"{len(df):,}")
    c2.metric("Segmentos", "4")
    c3.metric("Mejor Acción Global", f"{mejor_global[1]['ganador']}")
    c4.metric("Reward Total", f"{reward_total}")

    st.divider()

    col1, col2 = st.columns(2)

    conteo_segmentos = df["segmento"].value_counts()

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#1c1c1c')
        colores = [COLORES_SEGMENTO[s] for s in conteo_segmentos.index]
        wedges, texts, autotexts = ax.pie(
            conteo_segmentos.values,
            labels=conteo_segmentos.index,
            autopct='%1.1f%%',
            colors=colores,
            textprops={'color': '#f0f0f0'},
            wedgeprops={'edgecolor': '#1c1c1c', 'linewidth': 2}
        )
        ax.set_title("Distribución de Segmentos", color='#f0f0f0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#1c1c1c')
        estilizar_ejes(ax)
        colores = [COLORES_SEGMENTO[s] for s in conteo_segmentos.index]
        ax.bar(conteo_segmentos.index, conteo_segmentos.values, color=colores)
        ax.set_title("Clientes por Segmento", color='#f0f0f0')
        ax.set_ylabel("N° Clientes")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("### 🏆 Resumen de Ganadores — Thompson Sampling")

    resumen = pd.DataFrame([
        {
            "Segmento": f"{EMOJI_SEGMENTO[seg]} {seg}",
            "Promoción Ganadora": v["ganador"],
            "Conv. Rate": f"{v['conv_rate']}%",
            "Reward Acum.": v["reward"],
            "Regret Acum.": v["regret"],
            "Pulls Ganador": v["pulls_ganador"]
        }
        for seg, v in TS_DATA.items()
    ])
    st.table(resumen.set_index("Segmento"))

# =========================================================
# 2. SEGMENTACIÓN DE CLIENTES
# =========================================================
elif seccion == "👥 Segmentación de Clientes":
    st.markdown("## 👥 Segmentación de Clientes")
    st.caption("Perfil detallado de los 4 segmentos identificados vía K-Means Clustering")

    cols = st.columns(4)
    for col, seg in zip(cols, COLORES_SEGMENTO.keys()):
        sub = df[df["segmento"] == seg]
        with col:
            st.markdown(f"#### {EMOJI_SEGMENTO[seg]} {seg}")
            st.metric("Clientes", f"{len(sub):,}")
            st.metric("Edad Prom.", f"{sub['edad'].mean():.0f} años" if len(sub) else "—")
            st.metric("Gasto Prom.", f"S/.{sub['gasto_promedio_visita'].mean():.0f}" if len(sub) else "—")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Radar Comparativo de Segmentos")
        angles = np.linspace(0, 2 * np.pi, len(RADAR_DIMENSIONES), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6.5, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#1c1c1c')
        ax.set_facecolor('#1c1c1c')

        for seg, valores in RADAR_VALORES.items():
            vals = valores + valores[:1]
            ax.plot(angles, vals, linewidth=2, label=seg, color=COLORES_SEGMENTO[seg])
            ax.fill(angles, vals, alpha=0.08, color=COLORES_SEGMENTO[seg])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_DIMENSIONES, color='#f0f0f0', size=9)
        ax.set_yticklabels([])
        ax.tick_params(colors='#888888')
        ax.spines['polar'].set_color('#333333')
        ax.grid(color='#333333')
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), facecolor='#1c1c1c',
                  edgecolor='#333333', labelcolor='#f0f0f0', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("#### Frecuencia vs Gasto Promedio")
        sample = df.sample(n=min(2000, len(df)), random_state=42)

        fig, ax = plt.subplots(figsize=(6.5, 6))
        fig.patch.set_facecolor('#1c1c1c')
        estilizar_ejes(ax)
        for seg in COLORES_SEGMENTO.keys():
            sub = sample[sample["segmento"] == seg]
            ax.scatter(sub["frecuencia_visita_mes"], sub["gasto_promedio_visita"],
                       color=COLORES_SEGMENTO[seg], label=seg, alpha=0.6, s=18, edgecolors='none')
        ax.set_xlabel("Frecuencia de Visita (mes)")
        ax.set_ylabel("Gasto Promedio (S/.)")
        ax.legend(facecolor='#1c1c1c', edgecolor='#333333', labelcolor='#f0f0f0', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("#### 🔍 Explorar Clientes por Segmento")

    seleccion = st.multiselect(
        "Filtrar por segmento",
        options=list(COLORES_SEGMENTO.keys()),
        default=list(COLORES_SEGMENTO.keys())
    )

    columnas_tabla = [
        "id_cliente", "edad", "genero", "distrito", "segmento",
        "frecuencia_visita_mes", "gasto_promedio_visita", "nivel_fan_anime",
        "interacciones_instagram_mes", "valor_cliente_trim"
    ]

    tabla = df[df["segmento"].isin(seleccion)][columnas_tabla]
    st.dataframe(tabla, use_container_width=True, height=400)

# =========================================================
# 3. MOTOR THOMPSON SAMPLING
# =========================================================
elif seccion == "🎯 Motor Thompson Sampling":
    st.markdown("## 🎯 Motor Thompson Sampling")
    st.caption("Optimización de promociones vía Multi-Armed Bandit — 500 rondas por segmento")

    segmento_sel = st.selectbox("Selecciona un segmento", list(TS_DATA.keys()))
    data = TS_DATA[segmento_sel]
    color = COLORES_SEGMENTO[segmento_sel]

    st.markdown(
        f"""
        <div class="card-highlight" style="background-color:{color}22; border:1px solid {color};">
            <h2 style="color:{color};">{EMOJI_SEGMENTO[segmento_sel]} {segmento_sel} — Arm Ganador</h2>
            <p style="font-size:1.1rem;"><b>{data['ganador']}</b></p>
            <p>Conv. Rate: <b>{data['conv_rate']}%</b> &nbsp;|&nbsp;
               Reward: <b>{data['reward']}</b> &nbsp;|&nbsp;
               Pulls: <b>{data['pulls_ganador']}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_pulls = sum(a["pulls"] for a in data["arms"].values())
    filas = []
    for arm, v in data["arms"].items():
        p_media = v["alpha"] / (v["alpha"] + v["beta"]) * 100
        filas.append({
            "Arm": arm,
            "Pulls": v["pulls"],
            "% Pulls": f"{v['pulls'] / total_pulls * 100:.1f}%",
            "Rewards": v["rewards"],
            "Conv. Rate": f"{v['conv']}%",
            "α": v["alpha"],
            "β": v["beta"],
            "P(conv.) Media": f"{p_media:.1f}%",
            "Ganadora": "✅" if arm == data["ganador"] else ""
        })

    tabla_arms = pd.DataFrame(filas)
    st.dataframe(tabla_arms, use_container_width=True, hide_index=True)

    st.markdown("#### Probabilidad de Conversión Media por Arm")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('#1c1c1c')
    estilizar_ejes(ax)

    arms_list = list(data["arms"].keys())
    p_medias = [data["arms"][a]["alpha"] / (data["arms"][a]["alpha"] + data["arms"][a]["beta"]) * 100 for a in arms_list]
    colores_barras = ["#D4AF37" if a == data["ganador"] else "#555555" for a in arms_list]

    ax.barh(arms_list, p_medias, color=colores_barras)
    ax.set_xlabel("P(conv.) Media (%)")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# =========================================================
# 4. PERFORMANCE DEL MODELO
# =========================================================
elif seccion == "📈 Performance del Modelo":
    st.markdown("## 📈 Performance del Modelo")
    st.caption("Evolución de reward y regret del algoritmo Thompson Sampling")

    reward_total = sum(v["reward"] for v in TS_DATA.values())
    conversiones_totales = sum(a["rewards"] for v in TS_DATA.values() for a in v["arms"].values())
    total_pulls_global = sum(a["pulls"] for v in TS_DATA.values() for a in v["arms"].values())
    ctr_prom = conversiones_totales / total_pulls_global * 100
    regret_min = min(v["regret"] for v in TS_DATA.values())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reward Total", f"{reward_total}")
    c2.metric("Conversiones Totales", f"{conversiones_totales}")
    c3.metric("CTR Promedio", f"{ctr_prom:.1f}%")
    c4.metric("Regret Mínimo", f"{regret_min}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        fig.patch.set_facecolor('#1c1c1c')
        estilizar_ejes(ax)
        for seg, data in TS_DATA.items():
            ax.plot(data["convergencia"]["rondas"], data["convergencia"]["reward"],
                    marker='o', markersize=3, color=COLORES_SEGMENTO[seg], label=seg)
        ax.set_title("Reward Acumulado por Segmento", color='#f0f0f0')
        ax.set_xlabel("Rondas")
        ax.set_ylabel("Reward Acumulado")
        ax.legend(facecolor='#1c1c1c', edgecolor='#333333', labelcolor='#f0f0f0', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        fig.patch.set_facecolor('#1c1c1c')
        estilizar_ejes(ax)
        for seg, data in TS_DATA.items():
            ax.plot(data["convergencia"]["rondas"], data["convergencia"]["regret"],
                    marker='o', markersize=3, color=COLORES_SEGMENTO[seg], label=seg)
        ax.set_title("Regret Acumulado por Segmento", color='#f0f0f0')
        ax.set_xlabel("Rondas")
        ax.set_ylabel("Regret Acumulado")
        ax.legend(facecolor='#1c1c1c', edgecolor='#333333', labelcolor='#f0f0f0', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("#### Pulls por Segmento y Arm")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor('#1c1c1c')
    estilizar_ejes(ax)

    segmentos_list = list(TS_DATA.keys())
    bottom = np.zeros(len(segmentos_list))
    cmap_arms = plt.cm.get_cmap('tab10', len(ARMS))

    for i, arm in enumerate(ARMS):
        valores = [TS_DATA[seg]["arms"][arm]["pulls"] for seg in segmentos_list]
        ax.bar(segmentos_list, valores, bottom=bottom, label=arm, color=cmap_arms(i))
        bottom += np.array(valores)

    ax.set_ylabel("Pulls")
    ax.set_title("Distribución de Pulls por Arm", color='#f0f0f0')
    ax.legend(facecolor='#1c1c1c', edgecolor='#333333', labelcolor='#f0f0f0',
              fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.markdown("#### Tabla de Convergencia")
    seg_conv = st.selectbox("Selecciona un segmento", list(TS_DATA.keys()), key="conv_select")
    conv_data = TS_DATA[seg_conv]["convergencia"]
    tabla_conv = pd.DataFrame({
        "Ronda": conv_data["rondas"],
        "Reward Acumulado": conv_data["reward"],
        "Regret Acumulado": conv_data["regret"]
    })
    st.dataframe(tabla_conv, use_container_width=True, hide_index=True)

# =========================================================
# 5. INSIGHTS Y RECOMENDACIONES
# =========================================================
elif seccion == "💡 Insights y Recomendaciones":
    st.markdown("## 💡 Insights y Recomendaciones")
    st.caption("Estrategias accionables por segmento basadas en resultados del modelo")

    tabs = st.tabs([f"{EMOJI_SEGMENTO[s]} {s}" for s in ["Premium", "Otaku Core", "Digitales", "Ocasionales"]])

    for tab, seg in zip(tabs, ["Premium", "Otaku Core", "Digitales", "Ocasionales"]):
        with tab:
            info = INSIGHTS[seg]
            ts = TS_DATA[seg]
            color = COLORES_SEGMENTO[seg]
            n_clientes = len(df[df["segmento"] == seg])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clientes", f"{n_clientes:,}")
            c2.metric("Conv. Rate Ganadora", f"{ts['conv_rate']}%")
            c3.metric("Presupuesto Mensual", info["presupuesto"])
            c4.metric("Canal", info["canal"])

            st.info(f"💡 **Insight Thompson Sampling:** {info['insight']}\n\n📊 KPI esperado: {info['kpi']}")

            st.markdown("#### Tácticas Recomendadas")
            tabla_tacticas = pd.DataFrame(
                info["tacticas"],
                columns=["Táctica", "Descripción", "Canal", "Inversión", "ROI"]
            )
            st.dataframe(tabla_tacticas, use_container_width=True, hide_index=True)

            st.markdown(f"#### Distribución Beta — Arm Ganador ({ts['ganador']})")
            arm_data = ts["arms"][ts["ganador"]]
            a, b = arm_data["alpha"], arm_data["beta"]

            x = np.linspace(0, 1, 500)
            y = stats.beta.pdf(x, a, b)
            media = a / (a + b)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#1c1c1c')
            estilizar_ejes(ax)
            ax.plot(x, y, color=color, linewidth=2)
            ax.fill_between(x, y, color=color, alpha=0.2)
            ax.axvline(media, color='#f0f0f0', linestyle='--', linewidth=1.2,
                       label=f"Media = {media:.3f}")
            ax.set_xlabel("Tasa de Conversión")
            ax.set_ylabel("Densidad")
            ax.set_title(f"Beta(α={a}, β={b})", color='#f0f0f0')
            ax.legend(facecolor='#1c1c1c', edgecolor='#333333', labelcolor='#f0f0f0', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
