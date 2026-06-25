"""
calcul_trs.py
-------------
Fonctions de nettoyage et de calcul du Taux de Rendement Synthétique (TRS).

Le TRS se décompose en trois facteurs :
    Disponibilité = temps de marche / temps de production prévu
    Performance   = production réelle / production théorique
    Qualité       = pièces bonnes / pièces produites
    TRS           = Disponibilité x Performance x Qualité

Ces fonctions sont indépendantes de l'interface : elles peuvent être
utilisées dans un script, un notebook ou le tableau de bord Streamlit.
"""

import pandas as pd

# Colonnes attendues dans le fichier de production
COLONNES_REQUISES = [
    "machine", "date", "temps_prevu", "temps_marche",
    "prod_theorique", "prod_reelle", "pieces_bonnes",
]


def charger_donnees(chemin="production.csv"):
    """Lit le fichier CSV de production et convertit la date."""
    df = pd.read_csv(chemin)
    df["date"] = pd.to_datetime(df["date"])
    return df


def nettoyer(df):
    """
    Nettoie le jeu de données :
      - supprime les doublons exacts ;
      - supprime les lignes aux valeurs essentielles manquantes ;
      - écarte les lignes incohérentes (marche > prévu, bonnes > produites).
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=COLONNES_REQUISES)

    # Cohérences physiques
    df = df[df["temps_marche"] <= df["temps_prevu"]]
    df = df[df["pieces_bonnes"] <= df["prod_reelle"]]
    df = df[(df["temps_prevu"] > 0) &
            (df["prod_theorique"] > 0) &
            (df["prod_reelle"] > 0)]
    return df.reset_index(drop=True)


def calculer_trs(df):
    """
    Ajoute les colonnes dispo, perf, qual et trs (en proportion 0..1).
    Renvoie une copie du DataFrame enrichi.
    """
    df = df.copy()
    df["dispo"] = df["temps_marche"] / df["temps_prevu"]
    df["perf"] = df["prod_reelle"] / df["prod_theorique"]
    df["qual"] = df["pieces_bonnes"] / df["prod_reelle"]
    df["trs"] = df["dispo"] * df["perf"] * df["qual"]
    return df


def synthese_par_machine(df):
    """
    Renvoie la moyenne des composantes et du TRS par machine,
    triée du TRS le plus faible au plus élevé (pour repérer les priorités).
    """
    agg = (df.groupby("machine")[["dispo", "perf", "qual", "trs"]]
             .mean()
             .reset_index()
             .sort_values("trs"))
    return agg


def trs_global(df):
    """TRS moyen de l'ensemble de l'atelier (moyenne simple des relevés)."""
    return float(df["trs"].mean())


if __name__ == "__main__":
    # Petit test en ligne de commande
    data = charger_donnees()
    data = nettoyer(data)
    data = calculer_trs(data)
    print("TRS global de l'atelier : {:.1%}".format(trs_global(data)))
    print()
    syn = synthese_par_machine(data)
    for _, r in syn.iterrows():
        print("{} : TRS {:.1%}  (D {:.2f} | P {:.2f} | Q {:.2f})".format(
            r["machine"], r["trs"], r["dispo"], r["perf"], r["qual"]))
