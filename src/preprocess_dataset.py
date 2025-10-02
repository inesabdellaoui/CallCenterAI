import pandas as pd

from nltk.corpus import stopwords
import re
from langdetect import detect, LangDetectException
from sklearn.model_selection import train_test_split

# Étape 1 : Config stop words (mots inutiles)
stop_words_en = set(stopwords.words('english'))
stop_words_fr = set(stopwords.words('french'))  # Pour FR ; ajoutez 'arabic' si besoin via stop-words lib
stop_words = stop_words_en.union(stop_words_fr)  # Union pour multilingue

# Fonction nettoyage texte
def clean_text(text):
    if not isinstance(text, str):  
        return ''
    text = text.lower()  # Minuscules
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\d+', '', text)  
    text = ' '.join([word for word in text.split() if word not in stop_words])  
    return text.strip()  # espaces zeydin

# Fonction détection langue
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

# Étape 2 : Charger dataset brut
df = pd.read_csv('data/raw/all_tickets_processed_improved_v3.csv')

# Étape 3 : Nettoyage
df = df.dropna()  # Enlever manquants si présents
df['Document_clean'] = df['Document'].apply(clean_text)  # Applique nettoyage
df['Language'] = df['Document_clean'].apply(detect_language)  # Détecte langue

# Étape 4 : Vérifier après nettoyage
print("Distribution des langues détectées :\n", df['Language'].value_counts())  # Ex. : en: 90%, fr: 5%
print("Exemple ticket nettoyé :", df['Document_clean'].iloc[0])

# Étape 6 : Split en train/val/test (80/10/10, stratifié pour balance catégories)
train, temp = train_test_split(df, test_size=0.2, stratify=df['Topic_group'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['Topic_group'], random_state=42)

# Étape 7 : Sauvegarder fichiers processed
train.to_csv('data/processed/train.csv', index=False)
val.to_csv('data/processed/val.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)

print("Tailles après split : Train =", len(train), "Val =", len(val), "Test =", len(test))