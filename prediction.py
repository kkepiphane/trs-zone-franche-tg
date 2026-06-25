"""
prediction.py
-------------
Volet Big Data : maintenance predictive et prevision du TRS.

Deux modeles complementaires :

1) PREVISION DU TRS (regression lineaire) : a partir de l'historique
   journalier du TRS de chaque machine, on estime la tendance et on
   projette le TRS des prochains jours. Utile pour anticiper une derive.

2) PREDICTION DE RISQUE DE PANNE (Random Forest, classification) :
   on construit, par machine et par fenetre glissante, des indicateurs
   (features) decrivant l'etat recent de la machine, et on apprend a
   predire si une defaillance survient dans les jours suivants.
   On evalue le modele (train/test) et on fournit l'importance des features.

Tout est explicable, conforme a une demarche d'ingenieur :
features -> separation train/test -> entrainement -> evaluation -> interpretation.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

# Seuils metier
SEUIL_PANNE_TRS = 0.50      # un TRS journalier sous ce seuil = defaillance
FENETRE = 5                 # nb de jours d'historique pour construire les features
HORIZON = 5                 # on predit une panne dans les HORIZON jours suivants


# ============================================================
#  1) PREVISION DU TRS (tendance par regression lineaire)
# ============================================================
def prevoir_trs(df_machine, n_jours=15):
    """
    df_machine : DataFrame d'UNE machine, trie par date, avec colonne 'trs'.
    Renvoie (dates_futures, valeurs_prevues, pente_par_jour).
    """
    d = df_machine.sort_values("date").reset_index(drop=True)
    x = np.arange(len(d)).reshape(-1, 1)
    y = d["trs"].values

    modele = LinearRegression()
    modele.fit(x, y)

    x_futur = np.arange(len(d), len(d) + n_jours).reshape(-1, 1)
    y_futur = modele.predict(x_futur)
    y_futur = np.clip(y_futur, 0, 1)

    derniere_date = d["date"].max()
    dates_futures = pd.bdate_range(
        start=derniere_date + pd.Timedelta(days=1), periods=n_jours
    )
    pente = float(modele.coef_[0])   # variation de TRS par jour
    return dates_futures, y_futur, pente


# ============================================================
#  2) FEATURES pour la prediction de panne
# ============================================================
def construire_features(df):
    """
    Construit, pour chaque machine et chaque jour (a partir de FENETRE),
    des features decrivant l'etat RECENT, et une cible : panne dans les
    HORIZON jours suivants (1 = oui, 0 = non).

    Features (sur la fenetre glissante des FENETRE derniers jours) :
      - trs_moyen, trs_pente (tendance), dispo_moyenne
      - arrets_moyens, micro_moyens, temp_moyenne
    """
    lignes = []
    for machine, g in df.groupby("machine"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(FENETRE, len(g) - HORIZON):
            fenetre = g.iloc[i - FENETRE:i]
            futur = g.iloc[i:i + HORIZON]

            # tendance du TRS sur la fenetre (pente)
            xx = np.arange(FENETRE)
            pente = np.polyfit(xx, fenetre["trs"].values, 1)[0]

            cible = int((futur["trs"] < SEUIL_PANNE_TRS).any())

            lignes.append({
                "machine": machine,
                "date": g.iloc[i]["date"],
                "trs_moyen": fenetre["trs"].mean(),
                "trs_pente": pente,
                "dispo_moyenne": fenetre["dispo"].mean(),
                "arrets_moyens": fenetre["nb_arrets"].mean(),
                "micro_moyens": fenetre["micro_arrets"].mean(),
                "temp_moyenne": fenetre["temp_moteur"].mean(),
                "panne_future": cible,
            })
    return pd.DataFrame(lignes)


FEATURES = ["trs_moyen", "trs_pente", "dispo_moyenne",
            "arrets_moyens", "micro_moyens", "temp_moyenne"]


# ============================================================
#  3) ENTRAINEMENT + EVALUATION du modele de risque
# ============================================================
def entrainer_modele_risque(df_features):
    """
    Entraine un Random Forest pour predire 'panne_future'.
    Renvoie un dictionnaire avec le modele, les metriques, la matrice
    de confusion et l'importance des features.
    Gere le cas ou il n'y a qu'une seule classe (pas assez de signal).
    """
    X = df_features[FEATURES]
    y = df_features["panne_future"]

    resultat = {"ok": False, "raison": ""}

    if y.nunique() < 2:
        resultat["raison"] = (
            "Pas assez de variete dans les donnees (aucune panne detectee "
            "sur l'historique) pour entrainer un modele de classification.")
        return resultat

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    modele = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
    )
    modele.fit(X_tr, y_tr)
    y_pred = modele.predict(X_te)

    resultat.update({
        "ok": True,
        "modele": modele,
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "confusion": confusion_matrix(y_te, y_pred),
        "importances": pd.DataFrame({
            "feature": FEATURES,
            "importance": modele.feature_importances_,
        }).sort_values("importance", ascending=False),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    })
    return resultat


def risque_actuel_par_machine(df, df_features, modele):
    """
    Pour chaque machine, prend la fenetre la plus recente et predit
    la probabilite de panne dans les prochains jours.
    Renvoie un DataFrame trie par risque decroissant.
    """
    lignes = []
    for machine, g in df_features.groupby("machine"):
        derniere = g.sort_values("date").iloc[-1]
        proba = modele.predict_proba(derniere[FEATURES].to_frame().T)[0][1]
        lignes.append({"machine": machine, "probabilite_panne": float(proba)})
    res = pd.DataFrame(lignes).sort_values("probabilite_panne", ascending=False)

    def niveau(p):
        if p >= 0.60:
            return "ELEVE"
        if p >= 0.30:
            return "MODERE"
        return "FAIBLE"
    res["niveau"] = res["probabilite_panne"].map(niveau)
    return res


if __name__ == "__main__":
    from calcul_trs import charger_donnees, nettoyer, calculer_trs
    df = calculer_trs(nettoyer(charger_donnees("production.csv")))

    print("== Prevision TRS (M-03) ==")
    dfm = df[df["machine"] == "M-03"]
    dts, vals, pente = prevoir_trs(dfm, n_jours=15)
    print(f"Pente TRS/jour : {pente:+.4f}  -> TRS projete dans 15 j : {vals[-1]:.1%}")

    print("\n== Modele de risque de panne ==")
    feats = construire_features(df)
    print(f"Echantillons features : {len(feats)}  | pannes : {int(feats['panne_future'].sum())}")
    res = entrainer_modele_risque(feats)
    if res["ok"]:
        print(f"Train/Test : {res['n_train']}/{res['n_test']}")
        print(f"Accuracy {res['accuracy']:.2f} | Precision {res['precision']:.2f} "
              f"| Recall {res['recall']:.2f} | F1 {res['f1']:.2f}")
        print("Importance des features :")
        for _, r in res["importances"].iterrows():
            print(f"  {r['feature']:16s} {r['importance']:.3f}")
        risque = risque_actuel_par_machine(df, feats, res["modele"])
        print("\nRisque actuel par machine :")
        for _, r in risque.iterrows():
            print(f"  {r['machine']} : {r['probabilite_panne']:.0%}  [{r['niveau']}]")
    else:
        print("Modele non entraine :", res["raison"])


def evaluer_modele_charge(df_features, modele):
    """
    Evalue un modele DEJA entraine (charge depuis joblib) sur un jeu test
    reconstruit a partir des features. Renvoie le meme format que
    entrainer_modele_risque, pour un affichage homogene dans le dashboard.
    """
    X = df_features[FEATURES]
    y = df_features["panne_future"]
    resultat = {"ok": False, "raison": ""}
    if y.nunique() < 2:
        resultat["raison"] = "Pas assez de variete (aucune panne) pour evaluer."
        return resultat

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    y_pred = modele.predict(X_te)

    importances = None
    if hasattr(modele, "feature_importances_"):
        importances = pd.DataFrame({
            "feature": FEATURES, "importance": modele.feature_importances_,
        }).sort_values("importance", ascending=False)

    resultat.update({
        "ok": True, "modele": modele,
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "confusion": confusion_matrix(y_te, y_pred),
        "importances": importances,
        "n_train": len(X_tr), "n_test": len(X_te),
    })
    return resultat
