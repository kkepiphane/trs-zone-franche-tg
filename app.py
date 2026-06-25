"""
app.py — Tableau de bord TRS & Maintenance predictive (Master Big Data)
-----------------------------------------------------------------------
Chaine complete d'un projet Big Data, en 4 etapes (onglets) :

  1. Ingestion    : upload du CSV brut + validation du schema
  2. Preparation  : nettoyage + rapport de qualite des donnees
  3. Analyse TRS  : calcul, KPI, classement, decomposition, diagnostic
  4. Prediction   : prevision du TRS + risque de panne (Machine Learning)

Lancement :  streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from calcul_trs import (
    charger_donnees, nettoyer, calculer_trs,
    synthese_par_machine, trs_global, COLONNES_REQUISES,
)
import generer_donnees
import prediction as pred

st.set_page_config(
    page_title="TRS & Maintenance predictive",
    page_icon="https://img.icons8.com/ios-filled/50/1B2A4A/settings.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
#  Palette
# ---------------------------------------------------------------
NAVY   = "#1B2A4A"
SLATE  = "#3D5A80"
ACCENT = "#E07A5F"
GREEN  = "#2D6A4F"
RED    = "#C1292E"
AMBER  = "#D4A017"
LIGHT  = "#F0F2F6"
WHITE  = "#FFFFFF"
TEXT   = "#2C3E50"
MUTED  = "#7F8C8D"
SEUIL_EXCELLENCE = 0.85

# ---------------------------------------------------------------
#  CSS personnalise
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header hero */
.hero-header {
    background: linear-gradient(135deg, #1B2A4A 0%, #3D5A80 100%);
    padding: 2.5rem 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero-header h1 {
    color: #FFFFFF;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: rgba(255,255,255,0.75);
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: #fff;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    margin-top: 0.8rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #F0F2F6;
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 500;
    font-size: 0.88rem;
    color: #7F8C8D;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #1B2A4A !important;
    font-weight: 600;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* KPI card */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E8ECF1;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    transition: box-shadow 0.2s;
}
.kpi-card:hover {
    box-shadow: 0 4px 16px rgba(27,42,74,0.08);
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #7F8C8D;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1B2A4A;
    line-height: 1.1;
}
.kpi-delta {
    font-size: 0.8rem;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* Section title */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1B2A4A;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title .icon {
    font-size: 1rem;
    opacity: 0.7;
}

/* Content card */
.content-card {
    background: #FFFFFF;
    border: 1px solid #E8ECF1;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Risk card */
.risk-card {
    background: #FFFFFF;
    border: 1px solid #E8ECF1;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin: 0.5rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: transform 0.15s, box-shadow 0.15s;
}
.risk-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.risk-machine {
    font-weight: 600;
    color: #1B2A4A;
    font-size: 0.95rem;
}
.risk-badge {
    font-size: 0.78rem;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}

/* Qualite checklist */
.check-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.check-list li {
    padding: 0.5rem 0;
    border-bottom: 1px solid #F0F2F6;
    color: #2C3E50;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.check-list li:last-child { border-bottom: none; }
.check-icon {
    color: #2D6A4F;
    font-weight: 700;
    font-size: 0.85rem;
}

/* Diagnostic */
.diagnostic-card {
    background: linear-gradient(135deg, #FAFBFD, #F0F2F6);
    border: 1px solid #E8ECF1;
    border-left: 4px solid #E07A5F;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.diagnostic-card .title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #E07A5F;
    margin-bottom: 0.4rem;
}
.diagnostic-card .body {
    color: #2C3E50;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #7F8C8D;
    font-size: 0.8rem;
    border-top: 1px solid #E8ECF1;
    margin-top: 2rem;
}

/* Step badge */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #F0F2F6;
    color: #1B2A4A;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 8px;
    margin-bottom: 0.6rem;
}
.step-num {
    background: #1B2A4A;
    color: #fff;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
}

/* Streamlit overrides */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8ECF1;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: none;
}
div[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 16px rgba(27,42,74,0.06);
}
div[data-testid="stMetric"] label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #7F8C8D !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #1B2A4A !important;
}

.stDownloadButton > button,
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    font-size: 0.88rem;
    border: 1px solid #E8ECF1;
    transition: all 0.2s;
}

div[data-testid="stFileUploader"] {
    border-radius: 12px;
}

.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

.stDivider {
    margin: 1.5rem 0 !important;
}

hr {
    border-color: #E8ECF1 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------
def donnees_par_defaut():
    if not os.path.exists("production.csv"):
        generer_donnees.generer().to_csv("production.csv", index=False)
    return pd.read_csv("production.csv")


def couleur_trs(v):
    if v >= SEUIL_EXCELLENCE:
        return GREEN
    if v >= 0.70:
        return AMBER
    return RED


def chart_layout(fig, height=360):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(t=24, b=16, l=16, r=16),
        font=dict(family="Inter, sans-serif", size=12, color=TEXT),
        xaxis=dict(gridcolor="#F0F2F6", zerolinecolor="#E8ECF1"),
        yaxis=dict(gridcolor="#F0F2F6", zerolinecolor="#E8ECF1"),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
    )
    return fig


# ---------------------------------------------------------------
#  Hero Header
# ---------------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <h1>TRS & Maintenance Predictive</h1>
    <p>Optimisation de la production des societes industrielles
    de la Zone Franche au Togo</p>
    <span class="hero-badge">Chaine Big Data complete</span>
</div>
""", unsafe_allow_html=True)

if "df_brut" not in st.session_state:
    st.session_state.df_brut = donnees_par_defaut()

onglet1, onglet2, onglet3, onglet4 = st.tabs([
    "  Ingestion  ",
    "  Preparation  ",
    "  Analyse TRS  ",
    "  Prediction ML  ",
])

# ===============================================================
#  ONGLET 1 — INGESTION
# ===============================================================
with onglet1:
    st.markdown(
        '<div class="step-badge"><span class="step-num">1</span> Ingestion des donnees</div>',
        unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#7F8C8D;font-size:0.9rem'>"
        "Chargez le fichier de production d'une usine, ou utilisez le "
        "jeu de demonstration. Le schema est verifie automatiquement.</span>",
        unsafe_allow_html=True)

    st.write("")
    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        st.markdown('<div class="section-title"><span class="icon">&#x25B2;</span> Import</div>',
                    unsafe_allow_html=True)
        fichier = st.file_uploader("Deposer un fichier CSV de production",
                                   type=["csv"], label_visibility="collapsed")
        if fichier is not None:
            try:
                df_up = pd.read_csv(fichier)
                manquantes = [c for c in COLONNES_REQUISES if c not in df_up.columns]
                if manquantes:
                    st.error("Colonnes manquantes : " + ", ".join(manquantes))
                    st.info("Colonnes attendues : " + ", ".join(COLONNES_REQUISES))
                else:
                    st.session_state.df_brut = df_up
                    st.success(f"Fichier charge — {len(df_up)} lignes importees.")
            except Exception as e:
                st.error(f"Lecture impossible : {e}")

        if st.button("Utiliser le jeu de demonstration", type="secondary",
                      use_container_width=True):
            st.session_state.df_brut = donnees_par_defaut()
            st.success("Jeu de demonstration charge.")

    with col_b:
        st.markdown('<div class="section-title"><span class="icon">&#x25A0;</span> Format attendu</div>',
                    unsafe_allow_html=True)
        st.code(", ".join(COLONNES_REQUISES), language="text")
        modele = pd.DataFrame(columns=COLONNES_REQUISES)
        st.download_button(
            "Telecharger le modele CSV",
            modele.to_csv(index=False).encode("utf-8"),
            file_name="modele_production.csv", mime="text/csv",
            use_container_width=True)

    st.write("")
    st.markdown('<div class="section-title">Apercu des donnees brutes</div>',
                unsafe_allow_html=True)
    st.dataframe(st.session_state.df_brut.head(12),
                 use_container_width=True, hide_index=True)
    st.caption(f"{len(st.session_state.df_brut)} lignes chargees")

# ===============================================================
#  ONGLET 2 — PREPARATION
# ===============================================================
with onglet2:
    st.markdown(
        '<div class="step-badge"><span class="step-num">2</span> Preparation & qualite</div>',
        unsafe_allow_html=True)

    df_brut = st.session_state.df_brut.copy()
    manquantes = [c for c in COLONNES_REQUISES if c not in df_brut.columns]
    if manquantes:
        st.error("Donnees inexploitables : colonnes manquantes — "
                 + ", ".join(manquantes))
        st.stop()

    n0 = len(df_brut)
    df_brut["date"] = pd.to_datetime(df_brut["date"], errors="coerce")
    df_clean = nettoyer(df_brut)
    n1 = len(df_clean)
    rejets = n0 - n1

    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes recues", f"{n0:,}".replace(",", " "))
    c2.metric("Lignes valides", f"{n1:,}".replace(",", " "))
    c3.metric("Lignes rejetees", f"{rejets:,}".replace(",", " "),
              delta=f"-{(rejets / n0 * 100):.1f} %" if n0 else "0 %",
              delta_color="inverse")

    st.write("")
    st.markdown('<div class="section-title">Controles de qualite appliques</div>',
                unsafe_allow_html=True)
    st.markdown("""
<div class="content-card">
<ul class="check-list">
    <li><span class="check-icon">&#10003;</span> Suppression des doublons exacts</li>
    <li><span class="check-icon">&#10003;</span> Suppression des lignes aux valeurs essentielles manquantes</li>
    <li><span class="check-icon">&#10003;</span> Coherence : temps de marche &le; temps prevu</li>
    <li><span class="check-icon">&#10003;</span> Coherence : pieces bonnes &le; pieces produites</li>
    <li><span class="check-icon">&#10003;</span> Valeurs strictement positives pour les denominateurs</li>
</ul>
</div>
""", unsafe_allow_html=True)

    manq = df_brut[COLONNES_REQUISES].isna().sum()
    manq = manq[manq > 0]
    if len(manq):
        st.markdown('<div class="section-title">Valeurs manquantes detectees</div>',
                    unsafe_allow_html=True)
        st.dataframe(manq.rename("nb manquants").to_frame(),
                     use_container_width=True)
    else:
        st.success("Aucune valeur manquante sur les colonnes essentielles.")

    st.session_state.df_clean = calculer_trs(df_clean)
    st.caption("Les donnees nettoyees alimentent les onglets Analyse et Prediction.")

# ===============================================================
#  ONGLET 3 — ANALYSE TRS
# ===============================================================
with onglet3:
    st.markdown(
        '<div class="step-badge"><span class="step-num">3</span> Analyse du TRS</div>',
        unsafe_allow_html=True)

    if "df_clean" not in st.session_state:
        st.info("Passez d'abord par l'onglet Preparation.")
        st.stop()
    df = st.session_state.df_clean

    st.write("")
    fcol1, fcol2 = st.columns([1, 1.4], gap="large")
    with fcol1:
        machines = sorted(df["machine"].unique())
        sel = st.multiselect("Machines", machines, default=machines)
    with fcol2:
        dmin, dmax = df["date"].min().date(), df["date"].max().date()
        per = st.slider("Periode", min_value=dmin, max_value=dmax,
                        value=(dmin, dmax), format="DD/MM/YYYY")

    if not sel:
        st.warning("Selectionnez au moins une machine.")
        st.stop()
    dff = df[df["machine"].isin(sel)
             & (df["date"].dt.date >= per[0])
             & (df["date"].dt.date <= per[1])]
    if dff.empty:
        st.warning("Aucune donnee pour cette selection.")
        st.stop()

    st.write("")
    trs_val = trs_global(dff)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("TRS global", f"{trs_val:.1%}",
              delta=f"{(trs_val - SEUIL_EXCELLENCE):.1%} vs objectif")
    k2.metric("Disponibilite", f"{dff['dispo'].mean():.1%}")
    k3.metric("Performance", f"{dff['perf'].mean():.1%}")
    k4.metric("Qualite", f"{dff['qual'].mean():.1%}")

    st.write("")
    g, d = st.columns([1.1, 1], gap="medium")
    with g:
        st.markdown('<div class="section-title">TRS moyen par machine</div>',
                    unsafe_allow_html=True)
        syn = synthese_par_machine(dff)
        syn["pct"] = syn["trs"] * 100
        fig = px.bar(syn, x="machine", y="pct",
                     text=syn["pct"].map(lambda v: f"{v:.1f} %"))
        fig.update_traces(
            marker_color=[couleur_trs(v) for v in syn["trs"]],
            textposition="outside",
            marker_line_width=0,
        )
        fig.add_hline(y=85, line_dash="dash", line_color=SLATE,
                      annotation_text="Objectif 85 %",
                      annotation_font_color=SLATE)
        fig.update_layout(yaxis_title="TRS (%)", xaxis_title="",
                          yaxis_range=[0, 105], showlegend=False)
        chart_layout(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    with d:
        st.markdown('<div class="section-title">Decomposition</div>',
                    unsafe_allow_html=True)
        mf = st.selectbox("Machine", syn["machine"].tolist(), index=0,
                          label_visibility="collapsed")
        r = syn[syn["machine"] == mf].iloc[0]
        fac = pd.DataFrame({
            "Composante": ["Disponibilite", "Performance", "Qualite"],
            "Valeur": [r["dispo"], r["perf"], r["qual"]],
        })
        figf = px.bar(fac, x="Valeur", y="Composante", orientation="h",
                      text=fac["Valeur"].map(lambda v: f"{v:.0%}"),
                      color="Composante",
                      color_discrete_sequence=[SLATE, NAVY, ACCENT])
        figf.update_traces(textposition="outside", marker_line_width=0)
        figf.update_layout(xaxis_range=[0, 1.08], showlegend=False,
                           xaxis_title="", yaxis_title="")
        chart_layout(figf, 300)
        st.plotly_chart(figf, use_container_width=True)
        trs_color = couleur_trs(r["trs"])
        st.markdown(
            f"<div style='text-align:center;font-size:0.92rem;color:{MUTED}'>"
            f"TRS de <b style='color:{NAVY}'>{mf}</b> : "
            f"<b style='color:{trs_color};font-size:1.1rem'>{r['trs']:.1%}</b></div>",
            unsafe_allow_html=True)

    st.write("")
    syn = synthese_par_machine(dff)
    pire = syn.iloc[0]
    comp = {"la disponibilite": pire["dispo"], "la performance": pire["perf"],
            "la qualite": pire["qual"]}
    cause = min(comp, key=comp.get)
    st.markdown(
        f"""<div class="diagnostic-card">
            <div class="title">Diagnostic automatique</div>
            <div class="body">
                Machine la plus faible : <b>{pire['machine']}</b>
                (TRS <b>{pire['trs']:.1%}</b>).
                Composante penalisante : <b>{cause}</b> ({comp[cause]:.2f}).
                <br>Action prioritaire : agir sur {cause} de {pire['machine']}.
            </div>
        </div>""",
        unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-title">Evolution du TRS dans le temps</div>',
                unsafe_allow_html=True)
    evol = (dff.groupby([pd.Grouper(key="date", freq="W"), "machine"])["trs"]
              .mean().reset_index())
    evol["pct"] = evol["trs"] * 100
    figl = px.line(evol, x="date", y="pct", color="machine", markers=True)
    figl.add_hline(y=85, line_dash="dash", line_color=SLATE,
                   annotation_text="Objectif 85 %",
                   annotation_font_color=SLATE)
    figl.update_layout(yaxis_title="TRS (%)", xaxis_title="",
                       yaxis_range=[0, 105])
    figl.update_traces(line_width=2.5, marker_size=5)
    chart_layout(figl, 380)
    st.plotly_chart(figl, use_container_width=True)

# ===============================================================
#  ONGLET 4 — PREDICTION (ML)
# ===============================================================
with onglet4:
    st.markdown(
        '<div class="step-badge"><span class="step-num">4</span> Prediction (Machine Learning)</div>',
        unsafe_allow_html=True)

    if "df_clean" not in st.session_state:
        st.info("Passez d'abord par l'onglet Preparation.")
        st.stop()
    df = st.session_state.df_clean

    st.write("")
    st.markdown('<div class="section-title">A. Prevision du TRS</div>',
                unsafe_allow_html=True)

    p1, p2 = st.columns([1, 1], gap="large")
    with p1:
        msel = st.selectbox("Machine a analyser", sorted(df["machine"].unique()))
    with p2:
        n_jours = st.slider("Horizon de prevision (jours ouvres)", 5, 30, 15)

    dfm = df[df["machine"] == msel]
    dts, vals, pente = pred.prevoir_trs(dfm, n_jours=n_jours)

    hist = dfm.sort_values("date")
    figp = go.Figure()
    figp.add_trace(go.Scatter(
        x=hist["date"], y=hist["trs"] * 100,
        mode="lines", name="Historique",
        line=dict(color=NAVY, width=2.5)))
    figp.add_trace(go.Scatter(
        x=dts, y=vals * 100, mode="lines+markers",
        name="Prevision",
        line=dict(color=ACCENT, dash="dash", width=2.5),
        marker=dict(size=6)))
    figp.add_hline(y=85, line_dash="dot", line_color=GREEN,
                   annotation_text="Objectif 85 %", annotation_font_color=GREEN)
    figp.add_hline(y=50, line_dash="dot", line_color=RED,
                   annotation_text="Seuil critique 50 %", annotation_font_color=RED)
    figp.update_layout(yaxis_title="TRS (%)", xaxis_title="",
                       yaxis_range=[0, 105])
    chart_layout(figp, 400)
    st.plotly_chart(figp, use_container_width=True)

    tendance = "baissiere" if pente < 0 else "stable ou haussiere"
    t_color = RED if pente < 0 else GREEN
    st.markdown(
        f"<div class='content-card' style='text-align:center'>"
        f"<span style='color:{MUTED};font-size:0.82rem;text-transform:uppercase;"
        f"letter-spacing:0.5px;font-weight:500'>Tendance</span><br>"
        f"<span style='color:{t_color};font-size:1.3rem;font-weight:700'>"
        f"{tendance.capitalize()}</span>"
        f"<span style='color:{MUTED};font-size:0.88rem'> "
        f"({pente * 100:+.2f} pt/jour)</span><br>"
        f"<span style='font-size:0.9rem;color:{TEXT}'>TRS projete dans {n_jours} jours : "
        f"<b style='color:{NAVY}'>{vals[-1]:.1%}</b></span>"
        f"</div>",
        unsafe_allow_html=True)

    st.write("")
    st.markdown(
        '<div class="section-title">B. Risque de panne (Random Forest)</div>',
        unsafe_allow_html=True)
    st.caption("Construction de features — separation train/test "
               "— entrainement — evaluation — interpretation")

    feats = pred.construire_features(df)
    n_pannes = int(feats["panne_future"].sum())
    st.markdown(
        f"<div class='content-card'>"
        f"<b>{len(feats)}</b> echantillons construits dont "
        f"<b>{n_pannes}</b> situations de panne dans l'historique."
        f"</div>",
        unsafe_allow_html=True)

    modele_charge = None
    source_modele = ""
    if os.path.exists("modele_risque.joblib"):
        try:
            artefact = joblib.load("modele_risque.joblib")
            modele_charge = artefact["modele"]
            source_modele = "Modele charge depuis le notebook (modele_risque.joblib)."
        except Exception:
            modele_charge = None

    if modele_charge is not None:
        st.success(source_modele)
        res = pred.evaluer_modele_charge(feats, modele_charge)
    else:
        st.info("Modele entraine a la volee (notebook non execute).")
        res = pred.entrainer_modele_risque(feats)

    if not res["ok"]:
        st.warning("Modele non entraine : " + res["raison"])
    else:
        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Exactitude", f"{res['accuracy']:.0%}")
        m2.metric("Precision", f"{res['precision']:.0%}")
        m3.metric("Rappel", f"{res['recall']:.0%}")
        m4.metric("Score F1", f"{res['f1']:.0%}")
        st.caption(f"Entrainement sur {res['n_train']} echantillons, "
                   f"test sur {res['n_test']}.")

        if res["recall"] < 0.6:
            st.info("Les pannes sont rares dans l'historique "
                    "(classes desequilibrees), ce qui limite le rappel. Pistes "
                    "d'amelioration : plus d'historique, reequilibrage "
                    "(SMOTE), ou ajustement du seuil de decision.")

        st.write("")
        cg, cd = st.columns(2, gap="medium")
        with cg:
            st.markdown('<div class="section-title">Importance des variables</div>',
                        unsafe_allow_html=True)
            imp = res["importances"]
            if imp is not None:
                figi = px.bar(imp, x="importance", y="feature", orientation="h")
                figi.update_traces(marker_color=NAVY, marker_line_width=0)
                figi.update_layout(xaxis_title="", yaxis_title="",
                                   yaxis=dict(autorange="reversed"))
                chart_layout(figi, 320)
                st.plotly_chart(figi, use_container_width=True)
            else:
                st.caption("Non disponible pour ce modele.")
        with cd:
            st.markdown('<div class="section-title">Matrice de confusion</div>',
                        unsafe_allow_html=True)
            cm = res["confusion"]
            figc = px.imshow(cm, text_auto=True,
                             color_continuous_scale=["#F0F2F6", NAVY],
                             labels=dict(x="Predit", y="Reel", color="Nb"),
                             x=["Sain", "Panne"], y=["Sain", "Panne"])
            figc.update_layout(coloraxis_showscale=False)
            chart_layout(figc, 320)
            st.plotly_chart(figc, use_container_width=True)

        st.write("")
        st.markdown('<div class="section-title">Risque actuel par machine</div>',
                    unsafe_allow_html=True)
        risque = pred.risque_actuel_par_machine(df, feats, res["modele"])
        badge_styles = {
            "ELEVE":  f"background:{RED}11;color:{RED};border:1px solid {RED}33",
            "MODERE": f"background:{AMBER}11;color:{AMBER};border:1px solid {AMBER}33",
            "FAIBLE": f"background:{GREEN}11;color:{GREEN};border:1px solid {GREEN}33",
        }
        for _, r in risque.iterrows():
            style = badge_styles.get(r["niveau"], "")
            st.markdown(
                f"""<div class="risk-card">
                    <span class="risk-machine">{r['machine']}</span>
                    <span class="risk-badge" style="{style}">
                        {r['probabilite_panne']:.0%} — {r['niveau']}
                    </span>
                </div>""",
                unsafe_allow_html=True)

        st.caption("Les machines a risque ELEVE doivent faire l'objet d'une "
                   "maintenance preventive prioritaire.")

# ---------------------------------------------------------------
#  Footer
# ---------------------------------------------------------------
st.markdown("""
<div class="app-footer">
    TRS & Maintenance Predictive &mdash; UCAO-UUT &middot; Master Big Data
    &middot; Donnees simulees de facon realiste
</div>
""", unsafe_allow_html=True)
