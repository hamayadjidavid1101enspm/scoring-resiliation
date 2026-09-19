# =============================================================================
# train_model.py — Script d'entraînement final
# Scoring de résiliation — Assurance Auto
# (Sans les affichages exploratoires, comme demandé à l'étape 15)
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ─── PARTIE A — Chargement & choix des variables ────────────────────────────

df = pd.read_excel('data/dataset_assurance_ML.xlsx')
df.to_csv('data/dataset_assurance_ML.csv', index=False, encoding='utf-8-sig')
df = pd.read_csv('data/dataset_assurance_ML.csv', encoding='utf-8-sig')

TARGET = 'Résiliation'

num_cols = ['Âge', 'Salaire Annuel (€)', 'Prime Annuelle (€)', 'Ancienneté (mois)',
            'Coeff. Bonus-Malus', 'Nb Sinistres (3 ans)',
            'Montant Sinistres (€)', 'Score Risque (0-100)']

cat_cols = ['Type Contrat', 'Catégorie Prof.', 'Usage Véhicule', 'Dernier Sinistre']

X = df[num_cols + cat_cols]
y = df[TARGET]

# ─── PARTIE B — Pipeline, entraînement & évaluation ─────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])

# On retient uniquement le Random Forest (meilleur candidat à l'étape 9)
pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=10,
        class_weight='balanced', random_state=42))
])

pipeline.fit(X_train, y_train)

# ─── PARTIE C — Sauvegarde du modèle et métadonnées ─────────────────────────

# Création du dossier models/ s'il n'existe pas
os.makedirs('models', exist_ok=True)

# Sauvegarde du pipeline
joblib.dump(pipeline, 'models/pipeline_resiliation.pkl')

# Sauvegarde des métadonnées pour Streamlit
y_proba = pipeline.predict_proba(X_test)[:, 1]
meta = {
    'modele': 'Random Forest',
    'auc_test': round(float(roc_auc_score(y_test, y_proba)), 3),
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'num_ranges': {c: {'min': float(X[c].min()), 'max': float(X[c].max()),
                       'median': float(X[c].median())} for c in num_cols},
    'cat_values': {c: sorted(X[c].unique().tolist()) for c in cat_cols},
}

with open('models/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Entraînement terminé ! Modèle (.pkl) et métadonnées (.json) sauvegardés dans 'models/'.")
