# 🚗 Scoring de Résiliation — Assurance Auto

Application de Machine Learning qui prédit la probabilité de résiliation d'un client d'assurance automobile.

## Structure du projet

```
scoring_resiliation/
├── data/                          # Données
│   ├── dataset_assurance_ML.xlsx  # Dataset brut (500 clients × 27 colonnes)
│   └── dataset_assurance_ML.csv   # Version CSV
├── models/                        # Modèle entraîné
│   ├── pipeline_resiliation.pkl   # Pipeline scikit-learn (prétraitement + Random Forest)
│   └── metadata.json              # Métadonnées (bornes, modalités)
├── .streamlit/
│   └── config.toml                # Thème personnalisé
├── train_model.py                 # Script d'entraînement
├── app.py                         # Interface Streamlit
├── requirements.txt               # Dépendances
└── README.md
```

## Lancement en local

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Modèle

- **Algorithme** : Random Forest (300 arbres, max_depth=4)
- **Features** : 8 numériques + 4 catégorielles
- **Pipeline** : StandardScaler + OneHotEncoder + RandomForestClassifier
- **Performance** : AUC = 0.861 sur le jeu de test

