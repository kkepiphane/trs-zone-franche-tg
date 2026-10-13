"""
app.py — Tableau de bord TRS & Maintenance predictive (Plotly Dash)
-------------------------------------------------------------------
Chaine Big Data complete, de la donnee brute jusqu'a la prediction :

  0. Vue d'ensemble : KPI, alerte maintenance, graphiques cles
  1. Ingestion      : upload du CSV brut + validation du schema
  2. Preparation    : nettoyage + rapport de qualite des donnees
  3. Analyse TRS    : KPI, classement, decomposition, diagnostic, evolution
                      (graphiques lies : cliquer une machine la met en avant)
  4. Prediction     : risque de panne (Random Forest) + prevision du TRS

Les calculs reutilisent calcul_trs.py et prediction.py sans modification.

Lancement :  python app.py   ->  http://127.0.0.1:8050
Production : gunicorn app:server
"""

import base64
import io
import os
import uuid

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from calcul_trs import (
    nettoyer, calculer_trs, synthese_par_machine, trs_global, COLONNES_REQUISES,
)
import generer_donnees
import prediction as pred

# ---------------------------------------------------------------
#  Palette
# ---------------------------------------------------------------
NAVY   = "#1B2A4A"
SLATE  = "#3D5A80"
ACCENT = "#E07A5F"
GREEN  = "#2D6A4F"
RED    = "#C1292E"
AMBER  = "#C98A0B"
TEXT   = "#2C3E50"
MUTED  = "#7F8C8D"
GRID   = "#EEF1F5"
GHOST  = "#C9D1DC"
COULEURS_MACHINES = [NAVY, ACCENT, "#3D8FB6", GREEN, AMBER, "#8E6CB8", "#5C6B7A"]

SEUIL_EXCELLENCE = 0.85
COLONNES_PREDICTION = ["nb_arrets", "micro_arrets", "temp_moteur"]
NOMS_FEATURES = {
    "trs_moyen": "TRS moyen",
    "trs_pente": "Tendance du TRS",
    "dispo_moyenne": "Disponibilité moyenne",
    "arrets_moyens": "Arrêts / jour",
    "micro_moyens": "Micro-arrêts / jour",
    "temp_moyenne": "Température moteur",
}
NIVEAUX = {
    "ELEVE":  ("Élevé", RED),
    "MODERE": ("Modéré", AMBER),
    "FAIBLE": ("Faible", GREEN),
}
GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    # export PNG haute definition (bouton appareil photo du graphique)
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

# ---------------------------------------------------------------
#  Donnees : cache cote serveur, une entree par jeu de donnees
# ---------------------------------------------------------------
_DONNEES = {}   # cle -> (DataFrame brut, nom affiche)
_CACHE = {}     # cle -> resultats calcules (nettoyage, TRS, modele)


def donnees_demo():
    if not os.path.exists("production.csv"):
        generer_donnees.generer().to_csv("production.csv", index=False)
    return pd.read_csv("production.csv")


_DONNEES["demo"] = (donnees_demo(), "Données simulées")


def preparer(cle):
    """Nettoyage + calcul du TRS, effectues une seule fois par jeu de donnees."""
    if cle not in _DONNEES:
        cle = "demo"
    if cle in _CACHE:
        return _CACHE[cle]
    brut, nom = _DONNEES[cle]
    df = brut.copy()
    manq = df[COLONNES_REQUISES].isna().sum()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    clean = calculer_trs(nettoyer(df))
    _CACHE[cle] = {
        "cle": cle, "nom": nom, "brut": brut, "clean": clean,
        "n0": len(brut), "n1": len(clean), "manquantes": manq[manq > 0],
        "ml": None,
    }
    return _CACHE[cle]


def modele_ml(cle):
    """Features + modele de risque + risque actuel par machine (mis en cache)."""
    p = preparer(cle)
    if p["ml"] is not None:
        return p["ml"]
    df = p["clean"]
    ml = {"ok": False, "raison": "", "source": ""}
    absentes = [c for c in COLONNES_PREDICTION if c not in df.columns]
    if absentes:
        ml["raison"] = ("Colonnes nécessaires au modèle de risque absentes : "
                        + ", ".join(absentes) + ".")
        p["ml"] = ml
        return ml

    feats = pred.construire_features(df)
    if feats.empty:
        ml["raison"] = "Historique trop court pour construire les indicateurs."
        p["ml"] = ml
        return ml

    # Le modele exporte par le notebook a ete entraine sur le jeu de
    # demonstration : on ne l'utilise que pour ces donnees-la.
    modele = None
    if p["cle"] == "demo" and os.path.exists("modele_risque.joblib"):
        try:
            modele = joblib.load("modele_risque.joblib")["modele"]
        except Exception:
            modele = None
    if modele is not None:
        res = pred.evaluer_modele_charge(feats, modele)
        source = "Modèle exporté par le notebook (modele_risque.joblib)"
    else:
        res = pred.entrainer_modele_risque(feats)
        source = "Modèle entraîné sur les données chargées"

    ml = dict(res)
    ml["source"] = source
    ml["n_echantillons"] = len(feats)
    ml["n_pannes"] = int(feats["panne_future"].sum())
    if res["ok"]:
        ml["risque"] = pred.risque_actuel_par_machine(df, feats, res["modele"])
    p["ml"] = ml
    return ml


# ---------------------------------------------------------------
#  Formatage
# ---------------------------------------------------------------
def pct(v, d=1, signe=False):
    s = f"{v * 100:+.{d}f}" if signe else f"{v * 100:.{d}f}"
    return s.replace(".", ",") + " %"


def nombre(n):
    return f"{n:,}".replace(",", " ")


def ecart_objectif(v):
    return (f"{(v - SEUIL_EXCELLENCE) * 100:+.1f}".replace(".", ",")
            + " pts vs objectif 85 %")


def couleur_trs(v):
    if v >= SEUIL_EXCELLENCE:
        return GREEN
    if v >= 0.70:
        return AMBER
    return RED


def couleurs_par_machine(machines):
    return {m: COULEURS_MACHINES[i % len(COULEURS_MACHINES)]
            for i, m in enumerate(sorted(machines))}


# ---------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------
def style_fig(fig, height=340, legende=True):
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(t=36, b=8, l=8, r=8),
        font=dict(family="Inter, sans-serif", size=12, color=TEXT),
        showlegend=legende,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11), title=None),
        hoverlabel=dict(font_family="Inter, sans-serif", bgcolor="white"),
        separators=", ",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def ligne_objectif(fig):
    fig.add_hline(y=85, line_dash="dash", line_color=SLATE, line_width=1,
                  annotation_text="Objectif 85 %", annotation_position="top left",
                  annotation_font=dict(color=SLATE, size=11))


def fig_trs_machines(df, focus=None, height=340):
    syn = synthese_par_machine(df)
    couleurs = [couleur_trs(v) for v in syn["trs"]]
    opac = [1 if (focus is None or m == focus) else 0.35 for m in syn["machine"]]
    fig = go.Figure(go.Bar(
        x=syn["machine"], y=syn["trs"] * 100,
        text=[pct(v) for v in syn["trs"]], textposition="outside",
        marker=dict(color=couleurs, opacity=opac, line_width=0),
        customdata=syn["machine"],
        hovertemplate="<b>%{x}</b><br>TRS %{y:.1f} %<extra></extra>",
    ))
    ligne_objectif(fig)
    fig.update_layout(yaxis_title="TRS (%)", yaxis_range=[0, 108], bargap=0.35)
    style_fig(fig, height, legende=False)
    fig.update_xaxes(showgrid=False)
    return fig


def fig_evolution(df, focus=None, height=360):
    evol = (df.groupby([pd.Grouper(key="date", freq="W"), "machine"])["trs"]
              .mean().reset_index())
    couleurs = couleurs_par_machine(evol["machine"].unique())
    fig = go.Figure()
    # la machine mise en avant est tracee en dernier (au premier plan)
    ordre = sorted(evol["machine"].unique(), key=lambda m: m == focus)
    for m in ordre:
        e = evol[evol["machine"] == m]
        en_avant = focus is None or m == focus
        fig.add_trace(go.Scatter(
            x=e["date"], y=e["trs"] * 100, name=m, mode="lines+markers",
            line=dict(color=couleurs[m] if en_avant else GHOST,
                      width=3.2 if m == focus else (2.2 if en_avant else 1.5)),
            marker=dict(size=6 if en_avant else 4),
            customdata=[m] * len(e),
            hovertemplate=f"<b>{m}</b><br>%{{x|%d/%m/%Y}}<br>TRS %{{y:.1f}} %<extra></extra>",
        ))
    ligne_objectif(fig)
    fig.update_layout(yaxis_title="TRS (%)", yaxis_range=[0, 105],
                      xaxis_tickformat="%d/%m", hovermode="closest")
    return style_fig(fig, height)


def fig_decomposition(r, height=260):
    comp = ["Disponibilité", "Performance", "Qualité"]
    vals = [r["dispo"], r["perf"], r["qual"]]
    fig = go.Figure(go.Bar(
        x=vals, y=comp, orientation="h",
        text=[pct(v, 0) for v in vals], textposition="outside",
        marker=dict(color=[SLATE, NAVY, ACCENT], line_width=0),
        hovertemplate="%{y} : %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(xaxis_range=[0, 1.12], xaxis_tickformat=".0%",
                      yaxis=dict(autorange="reversed"), bargap=0.4)
    return style_fig(fig, height, legende=False)


def fig_prevision(dfm, dates, vals, height=380):
    hist = dfm.sort_values("date")
    fig = go.Figure()
    fig.add_vrect(x0=dates[0], x1=dates[-1], fillcolor=ACCENT, opacity=0.06,
                  line_width=0)
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["trs"] * 100, mode="lines", name="Historique",
        line=dict(color=NAVY, width=2),
        hovertemplate="%{x|%d/%m/%Y}<br>TRS %{y:.1f} %<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=vals * 100, mode="lines+markers", name="Prévision",
        line=dict(color=ACCENT, dash="dash", width=2.5), marker=dict(size=5),
        hovertemplate="%{x|%d/%m/%Y}<br>TRS prévu %{y:.1f} %<extra></extra>",
    ))
    ligne_objectif(fig)
    fig.add_hline(y=50, line_dash="dot", line_color=RED, line_width=1,
                  annotation_text="Seuil critique 50 %",
                  annotation_position="bottom left",
                  annotation_font=dict(color=RED, size=11))
    fig.update_layout(yaxis_title="TRS (%)", yaxis_range=[0, 105],
                      xaxis_tickformat="%d/%m")
    return style_fig(fig, height)


def fig_risque(risque, focus=None, height=None):
    r = risque.sort_values("probabilite_panne")
    couleurs = [NIVEAUX[n][1] for n in r["niveau"]]
    opac = [1 if (focus is None or m == focus) else 0.45 for m in r["machine"]]
    fig = go.Figure(go.Bar(
        x=r["probabilite_panne"], y=r["machine"], orientation="h",
        text=[f"{pct(p, 0)} · {NIVEAUX[n][0]}"
              for p, n in zip(r["probabilite_panne"], r["niveau"])],
        textposition="outside",
        marker=dict(color=couleurs, opacity=opac, line_width=0),
        customdata=r["machine"],
        hovertemplate="<b>%{y}</b><br>Risque de panne %{x:.0%}<extra></extra>",
    ))
    for seuil in (0.30, 0.60):
        fig.add_vline(x=seuil, line_dash="dot", line_color=GHOST, line_width=1)
    fig.update_layout(xaxis_range=[0, 1.25], xaxis_tickformat=".0%", bargap=0.35,
                      xaxis_tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1])
    return style_fig(fig, height or 70 + 46 * len(r), legende=False)


def fig_importances(imp, height=300):
    imp = imp.sort_values("importance")
    fig = go.Figure(go.Bar(
        x=imp["importance"], y=[NOMS_FEATURES.get(f, f) for f in imp["feature"]],
        orientation="h", marker=dict(color=NAVY, line_width=0),
        hovertemplate="%{y} : %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(bargap=0.35)
    return style_fig(fig, height, legende=False)


def fig_confusion(cm, height=300):
    fig = px.imshow(cm, text_auto=True, x=["Sain", "Panne"], y=["Sain", "Panne"],
                    color_continuous_scale=["#F4F6F9", NAVY],
                    labels=dict(x="Prédit", y="Réel", color="Nb"))
    fig.update_traces(textfont_size=16)
    fig.update_layout(coloraxis_showscale=False)
    return style_fig(fig, height, legende=False)


def graphe(fig, id_=None, cliquable=False):
    kwargs = {"id": id_} if id_ else {}
    return dcc.Graph(figure=fig, config=GRAPH_CONFIG,
                     className="graph cliquable" if cliquable else "graph",
                     **kwargs)


# ---------------------------------------------------------------
#  Composants
# ---------------------------------------------------------------
def kpi(label, valeur, sous_titre=None, couleur=None):
    return html.Div(className="kpi", children=[
        html.Div(label, className="kpi-label"),
        html.Div(valeur, className="kpi-value", style={"color": couleur} if couleur else None),
        html.Div(sous_titre, className="kpi-sub") if sous_titre else None,
    ])


def carte(titre, *enfants, sous_titre=None, classe=""):
    return html.Div(className=f"card {classe}", children=[
        html.Div(className="card-head", children=[
            html.H3(titre),
            html.Span(sous_titre, className="card-sub") if sous_titre else None,
        ]),
        *enfants,
    ])


def entete(titre, sous_titre, p):
    return html.Div(className="page-head", children=[
        html.Div([html.H1(titre), html.P(sous_titre)]),
        html.Span(className="data-badge", children=[
            html.Span(className="dot"), p["nom"]]),
    ])


def tableau(df, n=10):
    df = df.head(n)
    return html.Div(className="table-wrap", children=html.Table(className="table", children=[
        html.Thead(html.Tr([html.Th(c) for c in df.columns])),
        html.Tbody([html.Tr([html.Td(str(v)) for v in ligne])
                    for ligne in df.itertuples(index=False)]),
    ]))


def note(texte, ton="info"):
    return html.Div(texte, className=f"note note-{ton}")


def carte_alerte(df, top, horizon=30):
    """Alerte de la machine la plus a risque : message, chiffres cles, lien."""
    libelle, couleur = NIVEAUX[top["niveau"]]
    machine = top["machine"]
    _, vals, pente = pred.prevoir_trs(df[df["machine"] == machine], n_jours=horizon)
    sous_seuil = [i for i, v in enumerate(vals) if v < pred.SEUIL_PANNE_TRS]
    if sous_seuil:
        echeance = ["Son TRS devrait passer sous le seuil critique de 50 % dans ",
                    html.B(f"~{sous_seuil[0] + 1} jours ouvrés"), "."]
    else:
        echeance = ["Son TRS est orienté à la baisse." if pente < 0
                    else "Son TRS reste stable pour l'instant."]
    stats = [
        ("Risque de panne", pct(top["probabilite_panne"], 0), couleur),
        ("Tendance", pct(pente, 2, signe=True).replace(" %", " pt/j"),
         RED if pente < 0 else GREEN),
        ("TRS à 15 jours", pct(vals[min(14, len(vals) - 1)]), None),
    ]
    return html.Div(className="alert", style={"--alert": couleur}, children=[
        html.Div(className="alert-icon", children="!"),
        html.Div(className="alert-body", children=[
            html.Div(f"Alerte maintenance prédictive · risque {libelle.lower()}",
                     className="alert-kicker"),
            html.Div(className="alert-text", children=[
                html.B(machine), " risque une défaillance. ", *echeance]),
            html.Div("Action recommandée : planifier une maintenance préventive "
                     "avant la panne.", className="alert-action"),
        ]),
        html.Div(className="alert-stats", children=[
            html.Div(className="alert-stat", children=[
                html.Span(label),
                html.B(valeur, style={"color": c} if c else None),
            ]) for label, valeur, c in stats
        ]),
        dcc.Link("Voir la prédiction →", href="/prediction", className="btn"),
    ])


# ---------------------------------------------------------------
#  Pages
# ---------------------------------------------------------------
def page_vue_ensemble(cle):
    p = preparer(cle)
    df = p["clean"]
    if df.empty:
        return note("Aucune donnée exploitable après nettoyage.", "danger")
    ml = modele_ml(cle)
    trs = trs_global(df)
    debut, fin = df["date"].min(), df["date"].max()
    n_machines = df["machine"].nunique()

    syn = synthese_par_machine(df)
    pire = syn.iloc[0]
    sous_objectif = int((syn["trs"] < SEUIL_EXCELLENCE).sum())

    focus = None
    alerte = None
    n_eleve = "—"
    if ml["ok"]:
        risque = ml["risque"]
        n_eleve = str(int((risque["niveau"] == "ELEVE").sum()))
        top = risque.iloc[0]
        if top["niveau"] != "FAIBLE":
            focus = top["machine"]
            alerte = carte_alerte(df, top)
        else:
            alerte = html.Div(className="alert", style={"--alert": GREEN}, children=[
                html.Div(className="alert-icon", children="✓"),
                html.Div(className="alert-body", children=[
                    html.Div("Maintenance prédictive", className="alert-kicker"),
                    html.Div("Aucune machine ne présente de risque de panne élevé.",
                             className="alert-text"),
                ]),
            ])
    elif ml["raison"]:
        alerte = note("Modèle de risque indisponible : " + ml["raison"])

    facteurs = [("Disponibilité", df["dispo"].mean()),
                ("Performance", df["perf"].mean()),
                ("Qualité", df["qual"].mean())]
    def terme(nom, v, principal=False):
        couleur = couleur_trs(v) if principal and v < SEUIL_EXCELLENCE else None
        return html.Div(className="formula-term" + (" formula-main" if principal else ""), children=[
            html.Span(nom, className="formula-label"),
            html.Span(pct(v), className="formula-val", style={"color": couleur} if couleur else None),
            html.Div(className="formula-bar", children=[
                html.Div(className="formula-fill",
                         style={"width": f"{v * 100:.1f}%", "background": couleur} if couleur
                         else {"width": f"{v * 100:.1f}%"}),
                html.Div(className="formula-target", title="Objectif 85 %") if principal else None,
            ]),
        ])

    formule = [terme("TRS atelier", trs, principal=True),
               html.Span("=", className="formula-op")]
    for i, (nom, v) in enumerate(facteurs):
        if i:
            formule.append(html.Span("×", className="formula-op"))
        formule.append(terme(nom, v))

    etapes = [
        ("1", "Ingestion", "CSV d'usine, schéma validé", "/ingestion"),
        ("2", "Préparation", "Nettoyage et contrôles qualité", "/preparation"),
        ("3", "Analyse TRS", "KPI, diagnostic, évolution", "/analyse"),
        ("4", "Prédiction", "Random Forest et prévision", "/prediction"),
    ]
    return html.Div([
        html.Div(className="hero", children=[
            html.Div(className="hero-text", children=[
                html.Div("Maintenance prédictive · Zone Franche du Togo",
                         className="hero-kicker"),
                html.H1("Anticiper les pannes avant qu'elles n'arrêtent la production"),
                html.P(f"{n_machines} machines suivies du {debut:%d/%m/%Y} au "
                       f"{fin:%d/%m/%Y} · Python, Scikit-learn, Plotly Dash"),
            ]),
            html.Div(className="hero-panel", children=[
                html.Div(className="hero-panel-head", children=[
                    html.Span("Taux de rendement synthétique", className="hero-panel-label"),
                    html.Span("objectif 85 %", className="hero-target"),
                ]),
                html.Div(formule, className="formula"),
            ]),
        ]),
        html.Div(className="grid grid-4", children=[
            kpi("TRS global", pct(trs), ecart_objectif(trs),
                couleur_trs(trs)),
            kpi("Machine la plus faible", pire["machine"], f"TRS {pct(pire['trs'])}",
                couleur_trs(pire["trs"])),
            kpi("Sous l'objectif", f"{sous_objectif} / {n_machines}", "machines < 85 % de TRS"),
            kpi("Risque de panne élevé", f"{n_eleve} / {n_machines}",
                "machines · Random Forest", RED if n_eleve not in ("—", "0") else GREEN),
        ]),
        alerte,
        html.Div(className="grid grid-2", children=[
            carte("TRS moyen par machine", graphe(fig_trs_machines(df, focus))),
            carte("Évolution hebdomadaire du TRS", graphe(fig_evolution(df, focus, 340)),
                  sous_titre=f"{focus} mise en avant" if focus else None),
        ]),
        html.Div(className="pipeline", children=[
            dcc.Link(href=href, className="step", children=[
                html.Span(num, className="step-num"),
                html.Div([html.B(titre), html.Span(desc)]),
            ]) for num, titre, desc, href in etapes
        ]),
    ])


def page_ingestion(cle):
    p = preparer(cle)
    return html.Div([
        entete("Ingestion des données",
               "Chargez le fichier de production d'une usine ou utilisez le jeu de "
               "démonstration. Le schéma est vérifié automatiquement.", p),
        html.Div(className="grid grid-2", children=[
            carte("Import", dcc.Upload(
                id="upload", accept=".csv", className="upload",
                children=html.Div([
                    html.Div("⤒", className="upload-icon"),
                    html.Div([html.B("Déposez un fichier CSV"), " ou cliquez pour parcourir"]),
                    html.Div("Séparateur virgule ou point-virgule", className="muted"),
                ])),
                html.Button("Utiliser le jeu de démonstration", id="btn-demo",
                            className="btn btn-ghost", n_clicks=0),
                html.Div(id="ing-msg"),
            ),
            carte("Format attendu",
                  html.P("Colonnes obligatoires :", className="muted"),
                  html.Div([html.Code(c, className="chip-code") for c in COLONNES_REQUISES],
                           className="chip-row"),
                  html.P("Colonnes utilisées par le modèle de risque :", className="muted"),
                  html.Div([html.Code(c, className="chip-code") for c in COLONNES_PREDICTION],
                           className="chip-row"),
                  html.A("Télécharger le modèle CSV", className="btn btn-ghost",
                         download="modele_production.csv",
                         href="data:text/csv;charset=utf-8,"
                              + ",".join(COLONNES_REQUISES + COLONNES_PREDICTION)),
            ),
        ]),
        carte("Aperçu des données brutes", html.Div(id="ing-preview",
                                                     children=apercu(p))),
    ])


def apercu(p):
    return [tableau(p["brut"], 12),
            html.Div(f"{nombre(p['n0'])} lignes chargées · source : {p['nom']}",
                     className="muted small")]


def page_preparation(cle):
    p = preparer(cle)
    rejets = p["n0"] - p["n1"]
    controles = [
        "Suppression des doublons exacts",
        "Suppression des lignes aux valeurs essentielles manquantes",
        "Cohérence : temps de marche ≤ temps prévu",
        "Cohérence : pièces bonnes ≤ pièces produites",
        "Valeurs strictement positives pour les dénominateurs",
    ]
    manq = p["manquantes"]
    return html.Div([
        entete("Préparation & qualité",
               "Les données brutes sont nettoyées avant tout calcul de TRS.", p),
        html.Div(className="grid grid-3", children=[
            kpi("Lignes reçues", nombre(p["n0"])),
            kpi("Lignes valides", nombre(p["n1"]), couleur=GREEN),
            kpi("Lignes rejetées", nombre(rejets),
                pct(rejets / p["n0"]) + " du total" if p["n0"] else None,
                RED if rejets else None),
        ]),
        html.Div(className="grid grid-2", children=[
            carte("Contrôles de qualité appliqués", html.Ul(className="checks", children=[
                html.Li([html.Span("✓", className="check"), c]) for c in controles])),
            carte("Valeurs manquantes",
                  tableau(manq.rename("nb manquants").rename_axis("colonne").reset_index())
                  if len(manq) else
                  note("Aucune valeur manquante sur les colonnes essentielles.", "ok")),
        ]),
    ])


def page_analyse(cle):
    p = preparer(cle)
    df = p["clean"]
    if df.empty:
        return note("Aucune donnée exploitable après nettoyage.", "danger")
    machines = sorted(df["machine"].unique())
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    return html.Div([
        entete("Analyse du TRS",
               "Cliquez sur une machine (barre ou courbe) pour la mettre en avant "
               "dans tous les graphiques.", p),
        html.Div(className="filters card", children=[
            html.Div([html.Label("Machines"),
                      dcc.Checklist(id="f-machines", options=machines, value=machines,
                                    inline=True, className="chips")]),
            html.Div([html.Label("Période"),
                      dcc.DatePickerRange(id="f-dates", min_date_allowed=dmin,
                                          max_date_allowed=dmax, start_date=dmin,
                                          end_date=dmax, display_format="DD/MM/YYYY",
                                          first_day_of_week=1)]),
        ]),
        dcc.Store(id="focus", data=None),
        html.Div(id="a-kpis", className="grid grid-4"),
        html.Div(className="grid grid-12", children=[
            carte("TRS moyen par machine", graphe(go.Figure(), "g-machines", True)),
            carte("Décomposition", html.Div(id="a-decomp-titre", className="muted"),
                  graphe(go.Figure(), "g-decomp"), html.Div(id="a-decomp-trs")),
        ]),
        html.Div(id="a-diag"),
        carte("Évolution hebdomadaire du TRS", graphe(go.Figure(), "g-evol", True)),
    ])


def page_prediction(cle):
    p = preparer(cle)
    df = p["clean"]
    if df.empty:
        return note("Aucune donnée exploitable après nettoyage.", "danger")
    ml = modele_ml(cle)
    machines = sorted(df["machine"].unique())
    defaut = ml["risque"].iloc[0]["machine"] if ml["ok"] else machines[0]

    if ml["ok"]:
        bloc_risque = carte(
            "Risque de panne par machine",
            graphe(fig_risque(ml["risque"], defaut), "g-risque", True),
            html.Div("Cliquez sur une machine pour afficher sa prévision. "
                     "Seuils : modéré ≥ 30 %, élevé ≥ 60 %.", className="muted small"),
            sous_titre="Probabilité d'une défaillance (TRS < 50 %) "
                       f"dans les {pred.HORIZON} prochains jours",
        )
        metriques = [
            kpi("Exactitude", pct(ml["accuracy"], 0)),
            kpi("Précision", pct(ml["precision"], 0)),
            kpi("Rappel", pct(ml["recall"], 0)),
            kpi("Score F1", pct(ml["f1"], 0)),
        ]
        bloc_eval = html.Div([
            html.H2("Évaluation du modèle", className="section-title"),
            html.Div(f"{ml['source']} · {nombre(ml['n_echantillons'])} échantillons dont "
                     f"{ml['n_pannes']} situations de panne · entraînement {ml['n_train']} "
                     f"/ test {ml['n_test']}", className="muted small"),
            html.Div(className="grid grid-4", children=metriques),
            note("Les pannes sont rares dans l'historique (classes déséquilibrées), ce qui "
                 "limite le rappel. Pistes : plus d'historique, rééquilibrage (SMOTE), "
                 "ajustement du seuil de décision.") if ml["recall"] < 0.6 else None,
            html.Div(className="grid grid-2", children=[
                carte("Importance des variables",
                      graphe(fig_importances(ml["importances"]))
                      if ml["importances"] is not None else
                      html.Div("Non disponible pour ce modèle.", className="muted")),
                carte("Matrice de confusion", graphe(fig_confusion(ml["confusion"])),
                      sous_titre="Jeu de test"),
            ]),
        ])
    else:
        bloc_risque = note("Modèle de risque indisponible : " + ml["raison"], "warn")
        bloc_eval = None

    return html.Div([
        entete("Prédiction",
               "Random Forest pour le risque de panne, régression linéaire pour "
               "la tendance du TRS.", p),
        bloc_risque,
        html.Div(className="card", children=[
            html.Div(className="card-head card-head-controls", children=[
                html.H3("Prévision du TRS"),
                html.Div(className="controls", children=[
                    dcc.Dropdown(id="p-machine", options=machines, value=defaut,
                                 clearable=False, searchable=False, className="dd"),
                    html.Div(className="slider-wrap", children=[
                        html.Label("Horizon (jours ouvrés)"),
                        dcc.Slider(id="p-horizon", min=5, max=30, step=5, value=15,
                                   marks={i: str(i) for i in range(5, 31, 5)}),
                    ]),
                ]),
            ]),
            html.Div(className="grid grid-forecast", children=[
                graphe(go.Figure(), "g-prev"),
                html.Div(id="p-tendance", className="trend"),
            ]),
        ]),
        bloc_eval,
    ])


PAGES = {
    "/": ("Vue d'ensemble", "◎", page_vue_ensemble),
    "/ingestion": ("Ingestion", "1", page_ingestion),
    "/preparation": ("Préparation", "2", page_preparation),
    "/analyse": ("Analyse TRS", "3", page_analyse),
    "/prediction": ("Prédiction", "4", page_prediction),
}

# ---------------------------------------------------------------
#  Application
# ---------------------------------------------------------------
app = Dash(
    __name__,
    title="TRS & Maintenance prédictive",
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
    ],
)
server = app.server   # point d'entree WSGI pour le deploiement (gunicorn app:server)

app.layout = html.Div(className="shell", children=[
    dcc.Location(id="url"),
    dcc.Store(id="data-key", data="demo"),
    html.Aside(className="sidebar", children=[
        html.Div(className="brand", children=[
            html.Div("TRS", className="brand-logo"),
            html.Div([html.B("TRS & Maintenance"), html.Span("prédictive")]),
        ]),
        html.Div("Tableau de bord", className="nav-section"),
        html.Nav(className="nav", children=[
            dcc.Link(id={"type": "nav", "index": href}, href=href, className="nav-link",
                     children=[html.Span(icone, className="nav-icon"), titre])
            for href, (titre, icone, _) in PAGES.items()
        ]),
        html.Div(className="sidebar-foot", children=[
            html.B("Données simulées"),
            html.Span("Jeu réaliste généré pour la démonstration"),
            html.Span("UCAO-UUT · Master Big Data"),
        ]),
    ]),
    html.Main(id="page", className="main"),
])


@app.callback(Output("page", "children"),
              Input("url", "pathname"), State("data-key", "data"))
def afficher_page(chemin, cle):
    _, _, construire = PAGES.get(chemin, PAGES["/"])
    return construire(cle)


@app.callback(Output({"type": "nav", "index": ALL}, "className"),
              Input("url", "pathname"))
def nav_active(chemin):
    chemin = chemin if chemin in PAGES else "/"
    return ["nav-link active" if href == chemin else "nav-link" for href in PAGES]


# ---- Ingestion ----
def lire_csv(contenu):
    _, b64 = contenu.split(",", 1)
    brut = base64.b64decode(b64)
    for encodage in ("utf-8-sig", "latin-1"):
        try:
            texte = brut.decode(encodage)
            break
        except UnicodeDecodeError:
            continue
    # sep=None : detection automatique virgule / point-virgule (export Excel FR)
    return pd.read_csv(io.StringIO(texte), sep=None, engine="python")


@app.callback(Output("data-key", "data"), Output("ing-msg", "children"),
              Output("ing-preview", "children"),
              Input("upload", "contents"), Input("btn-demo", "n_clicks"),
              State("upload", "filename"), prevent_initial_call=True)
def charger(contenu, _clic, nom_fichier):
    if ctx.triggered_id == "btn-demo":
        return "demo", note("Jeu de démonstration chargé.", "ok"), apercu(preparer("demo"))
    if contenu is None:
        return no_update, no_update, no_update
    try:
        df = lire_csv(contenu)
    except Exception as e:
        return no_update, note(f"Lecture impossible : {e}", "danger"), no_update
    manquantes = [c for c in COLONNES_REQUISES if c not in df.columns]
    if manquantes:
        return no_update, note("Colonnes manquantes : " + ", ".join(manquantes), "danger"), no_update
    cle = uuid.uuid4().hex
    _DONNEES[cle] = (df, nom_fichier or "Fichier importé")
    p = preparer(cle)
    msg = f"{nom_fichier} chargé : {nombre(len(df))} lignes importées."
    absentes = [c for c in COLONNES_PREDICTION if c not in df.columns]
    if absentes:
        msg += " Modèle de risque indisponible (colonnes " + ", ".join(absentes) + " absentes)."
    return cle, note(msg, "ok"), apercu(p)


# ---- Analyse : graphiques lies ----
@app.callback(Output("focus", "data"),
              Input("g-machines", "clickData"), Input("g-evol", "clickData"),
              State("focus", "data"), prevent_initial_call=True)
def choisir_focus(clic_barre, clic_courbe, focus):
    clic = clic_barre if ctx.triggered_id == "g-machines" else clic_courbe
    if not clic:
        return no_update
    machine = clic["points"][0].get("customdata")
    if isinstance(machine, list):
        machine = machine[0]
    # un second clic sur la meme machine retire la mise en avant
    return None if machine == focus else machine


@app.callback(Output("a-kpis", "children"), Output("g-machines", "figure"),
              Output("a-decomp-titre", "children"), Output("g-decomp", "figure"),
              Output("a-decomp-trs", "children"), Output("a-diag", "children"),
              Output("g-evol", "figure"),
              Input("f-machines", "value"), Input("f-dates", "start_date"),
              Input("f-dates", "end_date"), Input("focus", "data"),
              State("data-key", "data"))
def maj_analyse(sel, debut, fin, focus, cle):
    df = preparer(cle)["clean"]
    vide = style_fig(go.Figure(), 120, legende=False)
    dff = df[df["machine"].isin(sel or [])]
    if debut:
        dff = dff[dff["date"] >= pd.to_datetime(debut)]
    if fin:
        dff = dff[dff["date"] <= pd.to_datetime(fin)]
    if dff.empty:
        msg = note("Aucune donnée pour cette sélection.", "warn")
        return msg, vide, "", vide, "", None, vide

    if focus not in dff["machine"].unique():
        focus = None
    trs = trs_global(dff)
    kpis = [
        kpi("TRS global", pct(trs),
            ecart_objectif(trs), couleur_trs(trs)),
        kpi("Disponibilité", pct(dff["dispo"].mean())),
        kpi("Performance", pct(dff["perf"].mean())),
        kpi("Qualité", pct(dff["qual"].mean())),
    ]

    syn = synthese_par_machine(dff)
    pire = syn.iloc[0]
    cible = syn[syn["machine"] == focus].iloc[0] if focus else pire
    titre = (f"Machine sélectionnée : {cible['machine']}" if focus
             else f"Machine la plus faible : {cible['machine']} (cliquez une autre barre)")
    trs_txt = html.Div(className="center", children=[
        "TRS de ", html.B(cible["machine"]), " : ",
        html.B(pct(cible["trs"]), style={"color": couleur_trs(cible["trs"]),
                                         "fontSize": "1.15rem"})])

    comp = {"la disponibilité": pire["dispo"], "la performance": pire["perf"],
            "la qualité": pire["qual"]}
    cause = min(comp, key=comp.get)
    diag = html.Div(className="diag", children=[
        html.Div("Diagnostic automatique", className="diag-title"),
        html.Div([
            "Machine la plus faible : ", html.B(pire["machine"]),
            f" (TRS {pct(pire['trs'])}). Composante pénalisante : ",
            html.B(cause), f" ({pct(comp[cause])}). ",
            html.Br(), f"Action prioritaire : agir sur {cause} de {pire['machine']}.",
        ]),
    ])
    return (kpis, fig_trs_machines(dff, focus, 360), titre,
            fig_decomposition(cible), trs_txt, diag, fig_evolution(dff, focus))


# ---- Prediction ----
@app.callback(Output("p-machine", "value"),
              Input("g-risque", "clickData"), prevent_initial_call=True)
def risque_vers_prevision(clic):
    if not clic:
        return no_update
    return clic["points"][0]["customdata"]


@app.callback(Output("g-risque", "figure"),
              Input("p-machine", "value"), State("data-key", "data"),
              prevent_initial_call=True)
def surligner_risque(machine, cle):
    return fig_risque(modele_ml(cle)["risque"], machine)


@app.callback(Output("g-prev", "figure"), Output("p-tendance", "children"),
              Input("p-machine", "value"), Input("p-horizon", "value"),
              State("data-key", "data"))
def maj_prevision(machine, n_jours, cle):
    df = preparer(cle)["clean"]
    dfm = df[df["machine"] == machine]
    dates, vals, pente = pred.prevoir_trs(dfm, n_jours=n_jours)
    baisse = pente < 0
    sous_seuil = [i for i, v in enumerate(vals) if v < pred.SEUIL_PANNE_TRS]
    if sous_seuil:
        alerte = html.Div(className="trend-alert", children=[
            "Seuil critique atteint dans ", html.B(f"~{sous_seuil[0] + 1} jours ouvrés")])
    elif baisse:
        alerte = html.Div("Seuil critique non atteint sur l'horizon", className="muted small")
    else:
        alerte = None

    dernier = dfm.sort_values("date")["trs"].tail(5).mean()
    tendance = [
        html.Div(machine, className="trend-machine"),
        html.Div("Tendance", className="kpi-label"),
        html.Div("Baissière" if baisse else "Stable ou haussière",
                 className="trend-value", style={"color": RED if baisse else GREEN}),
        html.Div(f"{pct(pente, 2, signe=True).replace(' %', ' pt')} / jour", className="muted"),
        html.Hr(),
        html.Div("TRS récent (5 j)", className="kpi-label"),
        html.Div(pct(dernier), className="trend-num"),
        html.Div(f"TRS projeté à {n_jours} jours", className="kpi-label"),
        html.Div(pct(vals[-1]), className="trend-num", style={"color": couleur_trs(vals[-1])}),
        alerte,
    ]
    return fig_prevision(dfm, dates, vals), tendance


if __name__ == "__main__":
    modele_ml("demo")   # calcule le modele avant la premiere visite
    app.run(debug=False, port=8050)
