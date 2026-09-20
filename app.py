# =============================================================================
# app.py — Interface Streamlit pour le scoring de résiliation
# Étapes 16 à 23 du TP
# =============================================================================

import json, joblib
import pandas as pd
import streamlit as st

# ─── ÉTAPE 17 : Configuration de la page (DOIT être le 1er appel Streamlit) ──
st.set_page_config(page_title='Scoring Résiliation', page_icon='🚗', layout='wide')

# ─── ÉTAPE 17 : Charger le modèle et les métadonnées (une seule fois) ────────
# @st.cache_resource : le .pkl et le .json ne sont lus qu'au 1er chargement
# Chaque clic de l'utilisateur ré-exécute le script, mais PAS cette fonction

@st.cache_resource
def charger_modele():
    pipeline = joblib.load('models/pipeline_resiliation.pkl')
    with open('models/metadata.json', encoding='utf-8') as f:
        meta = json.load(f)
    return pipeline, meta

pipeline, meta = charger_modele()
num_cols, cat_cols = meta['num_cols'], meta['cat_cols']
rng, cats = meta['num_ranges'], meta['cat_values']

# ─── ÉTAPE 16 : Titre et sous-titre ──────────────────────────────────────────
st.title('🚗 Scoring de résiliation — Assurance Auto')
st.caption(f"Modèle : {meta['modele']}  ·  AUC test : {meta['auc_test']}")

# ─── ÉTAPE 18 : Les curseurs des variables numériques ────────────────────────
# Chaque curseur est borné par le min/max du dataset (lu depuis metadata.json)
# et positionné sur la médiane — le profil "type" du portefeuille

st.sidebar.header('👤 Profil du client')

def curseur(col, step=1.0, fmt=None):
    r = rng[col]
    val = st.sidebar.slider(col, min_value=r['min'], max_value=r['max'],
                            value=r['median'], step=step, format=fmt)
    return int(val) if fmt == '%d' else val

client = {}
client['Âge']                   = curseur('Âge', 1.0, '%d')
client['Salaire Annuel (€)']    = curseur('Salaire Annuel (€)', 500.0, '%d')
client['Prime Annuelle (€)']    = curseur('Prime Annuelle (€)', 10.0, '%d')
client['Ancienneté (mois)']     = curseur('Ancienneté (mois)', 1.0, '%d')
client['Coeff. Bonus-Malus']    = curseur('Coeff. Bonus-Malus', 0.01, '%.2f')
client['Nb Sinistres (3 ans)']  = curseur('Nb Sinistres (3 ans)', 1.0, '%d')
client['Montant Sinistres (€)'] = curseur('Montant Sinistres (€)', 100.0, '%d')
client['Score Risque (0-100)']  = curseur('Score Risque (0-100)', 1.0, '%d')

# ─── ÉTAPE 19 : Les menus des variables catégorielles ────────────────────────
# Les modalités viennent du metadata : si on ré-entraîne avec de nouvelles
# catégories, l'interface se met à jour sans modifier app.py

st.sidebar.markdown('---')
for col in cat_cols:
    client[col] = st.sidebar.selectbox(col, cats[col])

# ─── ÉTAPE 20 : Bouton Prédire et résultat ───────────────────────────────────
# - On construit un DataFrame d'une ligne à partir des valeurs saisies
# - Le pipeline fait la prédiction (prétraitement + modèle)
# - On affiche la probabilité avec une jauge et un message coloré selon le seuil

st.sidebar.markdown('---')
st.sidebar.header('⚙️ Paramètres d\'analyse')
SEUIL_RISQUE = st.sidebar.number_input('Seuil de risque élevé (%)', min_value=0, max_value=100, value=55) / 100.0
SEUIL_MODERE = st.sidebar.number_input('Seuil de risque modéré (%)', min_value=0, max_value=100, value=40) / 100.0

if st.button('🔮 Prédire', type='primary', use_container_width=True):
    df_client = pd.DataFrame([client])[num_cols + cat_cols]
    proba = float(pipeline.predict_proba(df_client)[0, 1])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric('Probabilité de résiliation', f'{proba:.0%}')
        if proba >= SEUIL_RISQUE:
            st.error('⚠️ Client À RISQUE — action de rétention conseillée')
        elif proba >= SEUIL_MODERE:
            st.warning('🟠 Risque modéré — à surveiller')
        else:
            st.success('✅ Client fidèle — risque faible')
    with col2:
        st.write('Niveau de risque')
        st.progress(proba)
        st.write('Données envoyées au modèle :')
        st.dataframe(df_client.T.astype(str).rename(columns={0: 'Valeur'}),
                     use_container_width=True)

    # ─── ÉTAPE 21 : Importance des variables ─────────────────────────────────
    # Le Random Forest sait quelles variables comptent le plus pour ses décisions
    # On affiche les 8 plus influentes sous forme de graphique en barres

    model = pipeline.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        noms = pipeline.named_steps['prep'].get_feature_names_out()
        imp = (pd.Series(model.feature_importances_, index=noms)
                 .sort_values(ascending=False).head(8))
        imp.index = [n.split('__', 1)[1] for n in imp.index]
        st.subheader('📊 Les 8 variables les plus influentes du modèle')
        st.bar_chart(imp)
else:
    st.info('👈 Ajustez le profil dans la barre latérale, puis cliquez sur Prédire.')

