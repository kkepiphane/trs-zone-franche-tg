"""
generer_figures.py
------------------
Produit les figures (PDF vectoriel) du rapport technique LaTeX.

Usage (depuis la racine du projet) :
    python docs/rapport/generer_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))
from calcul_trs import charger_donnees, nettoyer, calculer_trs  # noqa: E402
import prediction as pred                                        # noqa: E402

SORTIE = Path(__file__).resolve().parent / "figures"
SORTIE.mkdir(exist_ok=True)

NAVY, SLATE, ACCENT = "#1B2A4A", "#3D5A80", "#E07A5F"
GREEN, RED, AMBER, GHOST = "#2D6A4F", "#C1292E", "#C98A0B", "#C9D1DC"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#B8C0CC",
    "axes.labelcolor": "#2C3E50",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6EAF0",
    "grid.linewidth": 0.6,
    "xtick.color": "#5B6676",
    "ytick.color": "#5B6676",
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


def sauver(fig, nom):
    fig.savefig(SORTIE / f"{nom}.pdf")
    plt.close(fig)
    print("ok", nom)


def couleur_trs(v):
    return GREEN if v >= 0.85 else (AMBER if v >= 0.70 else RED)


df = calculer_trs(nettoyer(charger_donnees(RACINE / "production.csv")))

# ---- 1. TRS moyen par machine (empile D, P, Q en tableau a cote) -------------
syn = df.groupby("machine")[["dispo", "perf", "qual", "trs"]].mean().sort_values("trs")
fig, ax = plt.subplots(figsize=(6.2, 2.6))
barres = ax.bar(syn.index, syn["trs"] * 100, color=[couleur_trs(v) for v in syn["trs"]],
                width=0.6, zorder=2)
for b, v in zip(barres, syn["trs"]):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v * 100:.1f} %".replace(".", ","),
            ha="center", fontsize=8.5, color="#2C3E50")
ax.axhline(85, color=SLATE, ls="--", lw=1)
ax.text(4.42, 86.5, "objectif 85 %", color=SLATE, fontsize=8, ha="right")
ax.set_ylim(0, 100)
ax.set_ylabel("TRS moyen (%)")
ax.grid(axis="x", visible=False)
sauver(fig, "trs_machines")

# ---- 2. Evolution hebdomadaire ------------------------------------------------
evol = (df.groupby([pd.Grouper(key="date", freq="W"), "machine"])["trs"].mean()
          .unstack())
fig, ax = plt.subplots(figsize=(6.2, 2.7))
for m in evol.columns:
    focus = m == "M-03"
    ax.plot(evol.index, evol[m] * 100, marker="o", ms=3 if focus else 2,
            lw=2.2 if focus else 1, color=RED if focus else GHOST,
            label=m, zorder=3 if focus else 2)
ax.axhline(85, color=SLATE, ls="--", lw=1)
ax.axhline(50, color=RED, ls=":", lw=1)
ax.text(evol.index[0], 86.5, "objectif 85 %", color=SLATE, fontsize=8)
ax.text(evol.index[0], 51.5, "seuil critique 50 %", color=RED, fontsize=8)
ax.set_ylim(40, 100)
ax.set_ylabel("TRS hebdomadaire (%)")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m"))
ax.legend(ncol=5, loc="lower left", fontsize=7.5, bbox_to_anchor=(0, -0.32))
sauver(fig, "evolution")

# ---- 3. Correlations ------------------------------------------------------------
cols = ["trs", "dispo", "perf", "qual", "nb_arrets", "micro_arrets", "temp_moteur"]
noms = ["TRS", "Dispo.", "Perf.", "Qualité", "Arrêts", "Micro-arrêts", "Temp. moteur"]
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(4.6, 3.8))
im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), noms, rotation=40, ha="right")
ax.set_yticks(range(len(cols)), noms)
ax.grid(False)
for i in range(len(cols)):
    for j in range(len(cols)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 else "#2C3E50")
fig.colorbar(im, fraction=0.046, pad=0.04)
sauver(fig, "correlations")

# ---- 4. Prevision M-03 ------------------------------------------------------------
m03 = df[df["machine"] == "M-03"].sort_values("date").reset_index(drop=True)
X = np.arange(len(m03)).reshape(-1, 1)
lr = LinearRegression().fit(X, m03["trs"])
dates_f, vals, pente = pred.prevoir_trs(m03, n_jours=15)
fig, ax = plt.subplots(figsize=(6.2, 2.8))
ax.plot(m03["date"], m03["trs"] * 100, color=NAVY, lw=1, label="TRS journalier observé")
ax.plot(m03["date"], lr.predict(X) * 100, color=SLATE, lw=1.2, ls="-.",
        label="droite de tendance (ajustement)")
ax.plot(dates_f, vals * 100, "o--", color=ACCENT, ms=2.5, lw=1.5,
        label="projection sur 15 jours ouvrés")
ax.axvspan(dates_f[0], dates_f[-1], color=ACCENT, alpha=0.07, lw=0)
ax.axhline(85, color=SLATE, ls="--", lw=0.9)
ax.axhline(50, color=RED, ls=":", lw=1)
ax.text(m03["date"].iloc[0], 51.5, "seuil critique 50 %", color=RED, fontsize=8)
ax.set_ylim(35, 95)
ax.set_ylabel("TRS de M-03 (%)")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m"))
ax.legend(loc="lower left", fontsize=7.5)
sauver(fig, "prevision_m03")

# ---- 5-7. Modele de risque -----------------------------------------------------------
F = pred.construire_features(df)
Xf, y = F[pred.FEATURES], F["panne_future"]
X_tr, X_te, y_tr, y_te = train_test_split(Xf, y, test_size=0.30, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42,
                            class_weight="balanced").fit(X_tr, y_tr)
proba = rf.predict_proba(X_te)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
for ax, seuil in zip(axes, (0.5, 0.2)):
    cm = confusion_matrix(y_te, (proba >= seuil).astype(int))
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("c", ["#F4F6F9", NAVY]),
              vmin=0, vmax=20)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > 10 else NAVY)
    ax.set_xticks([0, 1], ["Sain", "Panne"])
    ax.set_yticks([0, 1], ["Sain", "Panne"])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(f"seuil = {seuil}".replace(".", ","), fontsize=9)
    ax.grid(False)
fig.tight_layout()
sauver(fig, "confusion")

noms_f = {"trs_moyen": "TRS moyen", "trs_pente": "Pente du TRS",
          "dispo_moyenne": "Disponibilité moyenne", "arrets_moyens": "Arrêts / jour",
          "micro_moyens": "Micro-arrêts / jour", "temp_moyenne": "Température moteur"}
imp = pd.Series(rf.feature_importances_, index=pred.FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(5.2, 2.3))
ax.barh([noms_f[f] for f in imp.index], imp.values, color=NAVY, height=0.6, zorder=2)
for i, v in enumerate(imp.values):
    ax.text(v + 0.004, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=8)
ax.set_xlim(0, 0.36)
ax.set_xlabel("Importance (diminution moyenne de l'impureté de Gini)")
ax.grid(axis="y", visible=False)
sauver(fig, "importances")

seuils = np.linspace(0.05, 0.95, 91)
courbes = {
    "Précision": [precision_score(y_te, proba >= s, zero_division=1) for s in seuils],
    "Rappel": [recall_score(y_te, proba >= s, zero_division=0) for s in seuils],
    "F1": [f1_score(y_te, proba >= s, zero_division=0) for s in seuils],
}
fig, ax = plt.subplots(figsize=(5.4, 2.5))
for (nom, val), c, st in zip(courbes.items(), (SLATE, RED, NAVY), ("-", "-", "--")):
    ax.step(seuils, val, where="post", color=c, ls=st, lw=1.6, label=nom)
ax.axvline(0.5, color="#7F8C8D", lw=0.8, ls=":")
ax.axvline(0.2, color=GREEN, lw=0.8, ls=":")
ax.text(0.51, 0.08, "seuil par défaut 0,5", fontsize=7.5, color="#7F8C8D")
ax.text(0.21, 0.08, "0,2", fontsize=7.5, color=GREEN)
ax.set_xlabel("Seuil de décision sur la probabilité de panne")
ax.set_ylim(-0.03, 1.05)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
sauver(fig, "seuil")
