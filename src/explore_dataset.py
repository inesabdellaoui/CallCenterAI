import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Pour visualisations plus jolies

# Étape 1 : Charger le dataset brut
df = pd.read_csv('data/raw/all_tickets_processed_improved_v3.csv') 

# Étape 2 : Statistiques de base
print("Taille du dataset (lignes, colonnes) :", df.shape) 
print("Colonnes :", df.columns)  
print("Nombre de catégories uniques :", df['Topic_group'].nunique())  
print("Distribution des catégories (compte par catégorie) :\n", df['Topic_group'].value_counts())  # Pour voir si équilibré

# Étape 3 : Aperçu des données
print("Aperçu des 5 premières lignes :\n", df.head())
print("Exemple de ticket brut :", df['Document'].iloc[0])  # Un texte exemple
print("Longueur moyenne des tickets (en caractères) :", df['Document'].apply(len).mean())  # Ex. : 200-300 chars
print("Longueur min/max :", df['Document'].apply(len).min(), "/", df['Document'].apply(len).max())

# Étape 4 : Vérifier les problèmes
print("Valeurs manquantes par colonne :\n", df.isnull().sum())  
print("Tickets vides :", df[df['Document'].str.strip() == ''].shape[0])  # Si >0, notez pour nettoyage

# Étape 5 : Visualisations (sauvegardez pour rapport)
# Distribution catégories
plt.figure(figsize=(10, 6))
sns.countplot(y='Topic_group', data=df, order=df['Topic_group'].value_counts().index)
plt.title('Distribution des Catégories de Tickets')
plt.xlabel('Nombre de Tickets')
plt.ylabel('Catégories')
plt.savefig('data/processed/category_distribution.png')  # Image pour README ou rapport
plt.show()

# Histogramme longueurs textes
plt.figure(figsize=(10, 6))
sns.histplot(df['Document'].apply(len), bins=50)
plt.title('Distribution des Longueurs de Tickets')
plt.xlabel('Longueur en Caractères')
plt.ylabel('Fréquence')
plt.savefig('data/processed/text_lengths.png')
plt.show()

# Étape 6 : Insights manuels
# Ajoutez des prints personnalisés, ex. : pour imbalance, calculez pourcentage
category_percent = df['Topic_group'].value_counts(normalize=True) * 100
print("Pourcentages des catégories :\n", category_percent)