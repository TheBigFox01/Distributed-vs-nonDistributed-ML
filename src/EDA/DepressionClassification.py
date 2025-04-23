import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import re

def plot_categorical_feature(df, column_name):
    plt.figure(figsize=(8, 5))
    sea.countplot(x=column_name, data=df, palette='Set2')
    plt.title(f'Distribution of {column_name}')
    plt.xticks(rotation=45)
    plt.show()

def plot_numerical_feature(df, column_name):
    plt.figure(figsize=(8, 5))
    sea.histplot(df[column_name], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {column_name}')
    plt.show()

def plot_boxplot(df, column_name):
    plt.figure(figsize=(8, 5))
    sea.boxplot(x=df[column_name], palette='coolwarm')
    plt.title(f'Boxplot of {column_name}')
    plt.show()

import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import re
from imblearn.over_sampling import SMOTE

# 1. Caricamento e preparazione dei dati
cols = [
    'id', 'Gender', 'Age', 'City', 'Profession', 'Academic Pressure', 
    'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 
    'Sleep Duration', 'Dietary Habits', 'Degree', 'Have_you_ever_had_suicidal_thoughts?', 
    'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness', 'Depression'
]
df = pd.read_csv('Student Depression Dataset.csv', 
                 header=None, index_col=None, names=cols, na_values=[''])

df.columns = df.columns.str.replace('[() ]+', '_', regex=True)

# Conteggio valori null e drop

'''
sea.heatmap(df.isnull(), cmap="viridis")
plt.title("Presence of null data")
plt.show()
print("Numero di valori Null")
print(df.isnull().sum())
'''
#df.info()
#df.describe()
df.dropna(inplace=True)

# Rimozione colonne praticamente vuote

# Conversione dei valori di 'Job_Satisfaction' in numerici e gestione degli errori
df['Job_Satisfaction'] = pd.to_numeric(df['Job_Satisfaction'], errors='coerce')

df['Work_Pressure']= pd.to_numeric(df['Work_Pressure'], errors='coerce')

# Filtraggio per escludere righe con valori di 'Job_Satisfaction' diversi da 0.0
df = df[df['Job_Satisfaction'] == 0.0]

# Rimozione della colonna 'Job_Satisfaction' ora che non contiene variazioni
df.drop(columns=['Job_Satisfaction'], inplace=True)

# Filtraggio per mantenere solo le righe dove 'Profession' è 'Student'
df = df[df['Profession'] == 'Student']

# Rimozione della colonna 'Profession' poiché non è più necessaria
df.drop(columns=['Profession'], inplace=True)

# Filtraggio per escludere righe con valori di 'Job_Satisfaction' diversi da 0.0
df = df[df['Work_Pressure'] == 0.0]

# Rimozione della colonna 'Job_Satisfaction' ora che non contiene variazioni
df.drop(columns=['Work_Pressure'], inplace=True)

# EDA features categoriche
# 1.CITY
#plot_categorical_feature(df, "City")
#print(df['City'].value_counts())

# Controllo distribuzione delle città e filtraggio per città con almeno 30 campioni
city_counts = df['City'].value_counts()
threshold = 30
significant_cities = city_counts[city_counts >= threshold].index
df = df[df['City'].isin(significant_cities)]

city_categories = {
    'Kalyan': 0, 'Srinagar': 0, 'Hyderabad': 1, 'Vasai-Virar': 0, 'Lucknow': 1,
    'Thane': 0, 'Ludhiana': 0, 'Agra': 0, 'Surat': 1, 'Kolkata': 1, 'Jaipur': 1,
    'Patna': 0, 'Pune': 1, 'Visakhapatnam': 0, 'Ahmedabad': 1, 'Bhopal': 0, 
    'Chennai': 1, 'Meerut': 0, 'Rajkot': 0, 'Delhi': 1, 'Bangalore': 1, 
    'Ghaziabad': 0, 'Mumbai': 1, 'Vadodara': 0, 'Varanasi': 0, 'Nagpur': 0, 
    'Indore': 0, 'Kanpur': 0, 'Nashik': 0, 'Faridabad': 0
}

df['Is_Metropolis'] = df['City'].map(city_categories)

df.drop(columns=['City'], inplace=True)

# 2.DEGREE
#plot_categorical_feature(df, "Degree")
#print(df['Degree'].value_counts()

high_school = ['Class 12']
bachelors = ['B.Ed', 'B.Com', 'B.Arch', 'BCA', 'B.Tech', 'BHM', 'BSc', 'B.Pharm', 
             'BBA', 'MBBS', 'LLB', 'BE', 'BA']
masters_phd = ['MSc', 'MCA', 'M.Tech', 'M.Ed', 'M.Com', 'M.Pharm', 'MD', 'MBA', 
               'MA', 'PhD', 'LLM', 'MHM', 'ME']

def categorize_degree(degree):
    if degree in high_school:
        return 'high_school'
    elif degree in bachelors:
        return 'bachelor'
    elif degree in masters_phd:
        return 'masters_phd'
    else:
        return 'Altro'
    
df = df[df['Degree'] != 'Others']
df['Degree'] = df['Degree'].apply(categorize_degree)

# 3. Sleep_Duration
#plot_categorical_feature(df, "Sleep_Duration")
#print(df['Sleep_Duration'].value_counts()

# Funzione per estrarre il primo numero dalla stringa
def extract_number(value):
    match = re.search(r'\d+(\.\d+)?', value)  # Cerca numeri interi o decimali
    if match:
        return float(match.group(0))  # Restituisce il numero come float
    return None  # Se non c'è un numero, restituisce None

# Funzione per categorizzare la durata del sonno in base al numero estratto
def categorize_sleep_duration(value):
    number = extract_number(value)
    if number is not None:  # Se c'è un numero valido
        return 1 if number > 6 else 0  # 1 > di 6 ore, 0 per <= di 6 ore
    return None  # Se non c'è un numero valido

#Rimuovo le poche righe di Others che sono presenti nel dataset circa 18
df = df[df['Sleep_Duration'] != 'Others']
df['Sleep_Duration'] = df['Sleep_Duration'].apply(categorize_sleep_duration)

# 4. Dietary_Habits
#plot_categorical_feature(df,"Dietary_Habits")
#print(df['Dietary_Habits'].value_counts())
df = df[df['Dietary_Habits'] != "Others"]

# 5. Have_you_ever_had_suicidal_thoughts?', 'Family_History_of_Mental_Illness'
#plot_categorical_feature(df,"Have_you_ever_had_suicidal_thoughts?")
#plot_categorical_feature(df,"Family_History_of_Mental_Illness")
binary_columns = ['Have_you_ever_had_suicidal_thoughts?', 'Family_History_of_Mental_Illness']
for col in binary_columns:
    df[col] = df[col].replace({'Yes': 1, 'No': 0})

# 6. Gender
#plot_categorical_feature(df,"Gender")
#print(df['Gender'].value_counts())
df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})

# CREAZIONE DUMMY PER LE VARIABILI RIMASTE CATEGORICHE
df_encoded = pd.get_dummies(df, columns=['Dietary_Habits', 'Degree'], drop_first=False)

#Conversione delle collonne in numeri
columns_to_convert = ['id', 'Gender', 'Age', 'Academic_Pressure', 
    'CGPA', 'Study_Satisfaction', 'Sleep_Duration',
    'Have_you_ever_had_suicidal_thoughts?','Is_Metropolis', 
    'Work/Study_Hours', 'Financial_Stress', 'Family_History_of_Mental_Illness', 'Depression']

for col in columns_to_convert:
    try:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
        print(f"Colonna '{col}' convertita con successo in numerico.")
    except Exception as e:
        print(f"Errore durante la conversione della colonna '{col}': {e}")

# Conversione colonne Boolean in binarie
boolean_cols= boolean_columns = df_encoded.select_dtypes(include=['bool']).columns

for col in boolean_cols:
    df_encoded[col] = df_encoded[col].astype(int)

#EDA features numeriche

# 1. Study_Satisfaction
#plot_numerical_feature(df_encoded,"Study_Satisfaction")
#print(df_encoded['Study_Satisfaction'].value_counts())

df_encoded= df_encoded[df_encoded['Study_Satisfaction'] != 0.0]

#plot_numerical_feature(df_encoded,"Study_Satisfaction")
#print(df_encoded['Study_Satisfaction'].value_counts())

# 2. Academic_Pressure
#plot_numerical_feature(df_encoded,"Academic_Pressure")
#print(df_encoded['Academic_Pressure'].value_counts())
df_encoded = df_encoded[df_encoded['Academic_Pressure'] != 0.0]
#plot_numerical_feature(df_encoded,"Academic_Pressure")
#print(df_encoded['Academic_Pressure'].value_counts())

# 3. id
df_encoded.drop(columns=['id'], inplace=True)

# 4. Financial_Stress
#plot_numerical_feature(df_encoded,"Financial_Stress")
#print(df_encoded['Financial_Stress'].value_counts())

# 5. Age
#plot_numerical_feature(df_encoded,"Age")
#print(df_encoded['Age'].value_counts())

# 6. CGPA
#plot_numerical_feature(df_encoded,"CGPA")
#print(df_encoded['CGPA'].value_counts())

#OUTLIERS
non_binary_numeric_columns = ['Age' ,'CGPA']

'''
# Scatter Plot per ogni feature rispetto alle Labels
plt.figure(figsize=(15, 10))
for i, col in enumerate(non_binary_numeric_columns, 1):
    plt.subplot(3, 3, i)
    sea.scatterplot(x=col, y='Depression', data=df_encoded, alpha=0.7)
    plt.title(f"Scatter Plot: {col} vs Depression")
    plt.tight_layout()
plt.show()

# Box Plot per ogni feature rispetto alle Labels
plt.figure(figsize=(15, 10))
for i, col in enumerate(non_binary_numeric_columns, 1):
    plt.subplot(3, 3, i)
    sea.boxplot(x=df_encoded[col])
    plt.title(f"Box Plot: {col}")
    plt.tight_layout()
plt.show()
'''
# Rimuovere gli outlier con il metodo IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Rimozione degli outlier
for col in non_binary_numeric_columns:
    df_encoded = remove_outliers_iqr(df_encoded, col)

#CORRELAZIONE
columns_to_exclude = ['Depression']
# Creare un sottoinsieme del DataFrame escludendo le colonne specificate
df_corr_subset = df_encoded.drop(columns=columns_to_exclude, errors='ignore')

# Calcolare la matrice di correlazione sulle colonne selezionate
correlation_matrix = df_corr_subset.corr()

# Heatmap della matrice di correlazione
plt.figure(figsize=(12, 8))
sea.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matrice di correlazione")
plt.show()

# Salva il dataset modificato in un nuovo file CSV
df_encoded.to_csv('EDA_Student_Depression_Dataset.csv', index=False)
