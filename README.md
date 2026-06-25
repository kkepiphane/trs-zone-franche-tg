# Tableau de bord TRS & Maintenance predictive — Zone Franche Togo

Projet Big Data complet : de la donnee brute jusqu'a la **prediction**.
Calcul et suivi du **Taux de Rendement Synthetique (TRS)** + **maintenance
predictive** (prevision du TRS et prediction du risque de panne par machine).

```
TRS = Disponibilite x Performance x Qualite
```

## Apercu

Le tableau de bord Streamlit propose 4 pages accessibles via une barre laterale :

1. **Ingestion** — upload du CSV d'une usine + validation automatique du schema.
2. **Preparation** — nettoyage et rapport de qualite des donnees (lignes rejetees, etc.).
3. **Analyse TRS** — KPI, classement par machine, decomposition, diagnostic, evolution.
4. **Prediction (ML)** — prevision du TRS futur (regression) et risque de panne
   par machine (Random Forest), avec train/test, metriques et importance des variables.

## Prerequis

- **Python 3.10+**
- **pip**

## Installation

```bash
# 1. Cloner le depot
git clone https://github.com/kkepiphane/trs-zone-franche-tg.git
cd trs-zone-franche-tg

# 2. Creer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Installer les dependances
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

Le navigateur s'ouvre sur http://localhost:8501

> Les fichiers `production.csv` (donnees de demonstration) et `modele_risque.joblib`
> (modele ML) sont generes automatiquement au premier lancement si absents.

## Structure du projet

```
trs-zone-franche-tg/
├── app.py                              # Tableau de bord interactif (Streamlit)
├── calcul_trs.py                       # Nettoyage et calcul du TRS
├── generer_donnees.py                  # Generation de donnees simulees realistes
├── prediction.py                       # Modeles ML (prevision TRS + risque panne)
├── maintenance_predictive_TRS.ipynb    # Notebook d'analyse Data Science
├── requirements.txt                    # Dependances Python
├── .streamlit/
│   └── config.toml                     # Theme et configuration Streamlit
└── README.md
```

Fichiers generes (non versiones) :

| Fichier | Role |
|---|---|
| `production.csv` | Donnees de demonstration (~4 mois, 5 machines). Regenere par `generer_donnees.py`. |
| `modele_risque.joblib` | Modele Random Forest exporte. Regenere par le notebook ou a la volee par l'app. |

## Scenario de demonstration

- Atelier de 5 machines sur ~4 mois. TRS global ~79 %.
- **M-03** se degrade dans le temps : son TRS baisse, sa disponibilite chute.
- Page Prediction :
  - la **prevision** montre la tendance baissiere du TRS de M-03 ;
  - le **modele de risque** classe M-03 en risque ELEVE de panne (~90 %),
    les autres machines en risque FAIBLE.
- Message cle : on anticipe la defaillance et on planifie la maintenance AVANT la panne.

## Utiliser de vraies donnees

Remplacer le CSV par un fichier reel comportant au minimum les colonnes :

```
machine, date, temps_prevu, temps_marche, prod_theorique, prod_reelle, pieces_bonnes
```

Colonnes optionnelles exploitees par la prediction si presentes :
`nb_arrets, micro_arrets, temp_moteur`.

## Notebook d'analyse (demarche Data Science)

Le fichier **`maintenance_predictive_TRS.ipynb`** contient toute la demarche, etape
par etape : exploration (EDA), correlations, construction des features, entrainement
des deux modeles (prevision du TRS + risque de panne), evaluation et **export du
modele** (`modele_risque.joblib`).

### Ordre d'utilisation recommande

1. Generer les donnees : `python generer_donnees.py`
2. Ouvrir et executer le notebook : `jupyter notebook maintenance_predictive_TRS.ipynb`
   — il produit `modele_risque.joblib`.
3. Lancer le tableau de bord : `streamlit run app.py`
   — il **charge** le modele exporte par le notebook (et le reentraine seulement
   s'il ne le trouve pas).

Cette separation reproduit la pratique professionnelle : on **experimente et entraine**
dans le notebook, on **deploie** le modele fige dans l'application.

## Demarche d'ingenieur

- Pipeline complet visible : ingestion -> preparation -> analyse -> prediction.
- Construction de **features** (tendance du TRS, arrets, temperature moteur...).
- Separation **train/test**, **metriques** (exactitude, precision, rappel, F1),
  **matrice de confusion**, **importance des variables**.
- Honnetete scientifique : les pannes sont rares (classes desequilibrees), ce qui
  limite le rappel ; le dashboard l'explique et propose des pistes (plus de donnees,
  reequilibrage SMOTE, ajustement du seuil).

## Technologies

- Python, Pandas, Scikit-learn, Plotly, Streamlit, Joblib
