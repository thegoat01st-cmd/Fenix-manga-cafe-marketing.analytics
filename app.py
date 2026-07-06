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

# =========================================================
# BASE DE DATOS EMBEBIDA (10,000 clientes, comprimida gzip+base64)
# =========================================================
EMBEDDED_DATA_B64 = (
"H4sIAFX5SmoC/4y9y65tSXIkNu+vqA/YOIiHezyGYkMCWiClhhrQtHCZdVm6jay8RGaRf8WBppryxxRu7vFca+1zgEazsirX2esR4eFubm724y9//uXXH99/"
"+/v31/e/fPvL66/ff/v++8/XX3788ffff/z95+tffv/+y799/+2XH9/+/O8//vjx929//tv3P15//fbH33/++V9///m373/58dP+l9dvP/79+69//pdvv/35"
"228//vb99aP91d+//fLLj5+/ff/jzz9+++Pv3/76+7e/4S/8+7dff/7ef/rP7bf+9vrXn7//0v7p2//8/udffv7tX3//9gd+4adcj7/1m/ynP7cbwv/6+uPb"
"33/88S/69/+M+/7263/xzvlXcK//7fvfvv/247efr//7x6+/fvvT//rrn/7Ht1///dtffv7+olcMHzm+8suHD/eKtf1TepX8If9V+nDyN8IrhHd/g1/Rf8Tw"
"Kq+Q29+gVD/Ky7X/xP0vxJfP8y/8t9/+8v1fv7f/T95k+5ngP5L8nFxCTB8hvmr7fe/a/9Pr6eXT65++/fHLv/16/QPtEfmj/e/l5X37C5k/qLxSbPeQX1mv"
"5/366yNE+RvtZbWbaX8iyD/UV/Efuf1V+xupPd3yN/7Hv/3+y095ffRRY3tSj7v3Ve6E5DWkF+uF7W/W9cJvv/3pH37+/j/lxlP48CQ3Tu0CX9IHBXmX8l8V"
"vbjI99sv/qdvv//9P//jtz/95fuf/vvP339v60c+svyh9sZYftnJ30nlw8+7r6+wvoF/+vH7t39pi65dTHJxbB+7/Rvy7EQf1P67j9Reht5E+7Ptk8yL/+Hb"
"779/+609ffvw5SNy+7+hyOdurz6wvJHlWv/ydX76//r/tDtub//nH+0ikidlffgU5Nuxa5f2VeNDeyWv/+Pnn/75x2/ffv9hD//jr//2/deX/FWsE3wvzh+p"
"vti3T57sgX2UBx6/+7+0LR1e1P5lkm8m3ypw+Aj0yvKTtd9te3B/vO7/9sePv7Qg0P5wkqcrerWPOXywXNb+m6oXt9fAy6P+/Ns3e0ovLxdfxqd2UbuN9nn6"
"U6brT16/MMmycPI9g/y6/Hh7PW2dyevSddZ2WFg22T9++9M//Wx/8Vt7OIryfuQz4tZ9rrLhIrd7GU9e9tvod1/avcrPerl9H2L8aI/Z3gDNL9y+QdnfNNZA"
"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"
)

SHEET_OBJETIVO = "fenix_manga_cafe_10k"

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
# La base de datos (10,000 clientes) está embebida directamente en este archivo
# (comprimida en Base64) para que la app no dependa de ningún Excel externo.
# =========================================================
@st.cache_data
def cargar_datos():
    import gzip, base64, io

    csv_bytes = gzip.decompress(base64.b64decode(EMBEDDED_DATA_B64))
    df = pd.read_csv(io.BytesIO(csv_bytes))
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
