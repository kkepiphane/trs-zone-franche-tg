"""
generer_donnees.py
-------------------
Genere un jeu de donnees de production simule de facon realiste pour
le calcul du TRS et pour la maintenance predictive.

Atelier de 5 machines (M-01 a M-05), releves journaliers sur ~4 mois.

Realisme ajoute pour le volet Big Data / prediction :
  - chaque machine a un profil de base (dispo, perf, qual) ;
  - le nombre d'arrets et les micro-arrets sont enregistres (signaux d'usure) ;
  - certaines machines se DEGRADENT progressivement dans le temps
    (tendance baissiere de la disponibilite + hausse des arrets),
    ce qui cree un signal apprenable par un modele predictif ;
  - une panne est marquee quand la disponibilite chute sous un seuil.

Colonnes produites :
  machine, date, temps_prevu, temps_marche, prod_theorique, prod_reelle,
  pieces_bonnes, nb_arrets, micro_arrets, temp_moteur

Usage :
    python generer_donnees.py  -> ecrit "production.csv"
"""

import numpy as np
import pandas as pd

np.random.seed(42)

MACHINES = ["M-01", "M-02", "M-03", "M-04", "M-05"]

# Profil de base : (dispo, perf, qual) au debut de la periode
PROFILS = {
    "M-01": {"dispo": 0.93, "perf": 0.94, "qual": 0.99},
    "M-02": {"dispo": 0.90, "perf": 0.93, "qual": 0.97},
    "M-03": {"dispo": 0.86, "perf": 0.95, "qual": 0.98},
    "M-04": {"dispo": 0.91, "perf": 0.91, "qual": 0.96},
    "M-05": {"dispo": 0.94, "perf": 0.96, "qual": 0.99},
}

# Derive journaliere de la disponibilite (negatif = degradation).
DERIVE_DISPO = {
    "M-01": +0.00002,
    "M-02": -0.00010,
    "M-03": -0.00320,   # forte degradation -> devient critique (pannes)
    "M-04": -0.00150,   # degradation moderee -> quelques pannes
    "M-05": +0.00001,
}

DATE_DEBUT = "2026-02-02"
NB_JOURS_OUVRES = 90

TEMPS_PREVU_JOUR = 450
CADENCE_NOMINALE = 1200

SEUIL_PANNE_DISPO = 0.55


def _bornes(x, mini=0.40, maxi=0.999):
    return float(np.clip(x, mini, maxi))


def _clip_int(x, mini=0, maxi=60):
    return int(np.clip(x, mini, maxi))


def generer():
    dates = pd.bdate_range(start=DATE_DEBUT, periods=NB_JOURS_OUVRES)
    lignes = []

    for machine in MACHINES:
        p = PROFILS[machine]
        derive = DERIVE_DISPO[machine]

        for i, d in enumerate(dates):
            dispo_cible = p["dispo"] + derive * i
            dispo = _bornes(np.random.normal(dispo_cible, 0.04))
            perf = _bornes(np.random.normal(p["perf"], 0.035))
            qual = _bornes(np.random.normal(p["qual"], 0.012))

            temps_prevu = TEMPS_PREVU_JOUR
            temps_marche = round(temps_prevu * dispo)
            prod_theorique = round(CADENCE_NOMINALE * (temps_marche / temps_prevu))
            prod_reelle = round(prod_theorique * perf)
            pieces_bonnes = round(prod_reelle * qual)

            perte_dispo = max(0.0, 0.95 - dispo)
            nb_arrets = _clip_int(np.random.poisson(1 + perte_dispo * 18))
            micro_arrets = _clip_int(np.random.poisson(2 + perte_dispo * 25))
            temp_moteur = round(np.random.normal(65 + perte_dispo * 40, 3), 1)

            lignes.append({
                "machine": machine,
                "date": d.strftime("%Y-%m-%d"),
                "temps_prevu": temps_prevu,
                "temps_marche": temps_marche,
                "prod_theorique": prod_theorique,
                "prod_reelle": prod_reelle,
                "pieces_bonnes": pieces_bonnes,
                "nb_arrets": nb_arrets,
                "micro_arrets": micro_arrets,
                "temp_moteur": temp_moteur,
            })

    return pd.DataFrame(lignes)


if __name__ == "__main__":
    df = generer()
    df.to_csv("production.csv", index=False)
    print(f"Fichier 'production.csv' genere : {len(df)} lignes "
          f"({df['machine'].nunique()} machines, {df['date'].nunique()} jours).")
