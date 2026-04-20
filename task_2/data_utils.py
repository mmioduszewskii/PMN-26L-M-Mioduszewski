import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

def load_heart_data():
    # pobranie danych
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    df = X.copy()
    df['target'] = y
    
    print(f"Dane załadowane pomyślnie. Kształt zbioru: {df.shape}")
    return df

def analyze_missing_data(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nBrakujące dane:")
    print(missing if not missing.empty else "Brak brakujących danych!")
    return missing

def plot_eda(df):
    fig = plt.figure(figsize=(16, 10))
    
    # rozkład klas
    plt.subplot(2, 2, 1)
    sns.countplot(x='target', data=df, palette='Set2')
    plt.title('Rozkład zmiennej docelowej (0 = brak, 1-4 = stadium choroby)')
    
    # rozkład wieku
    plt.subplot(2, 2, 2)
    sns.histplot(df['age'], kde=True, bins=20, color='skyblue')
    plt.title('Rozkład wieku pacjentów')
    
    # rozkład maksymalnego tętna
    plt.subplot(2, 2, 3)
    sns.histplot(df['thalach'], kde=True, bins=20, color='salmon')
    plt.title('Rozkład maksymalnego osiągniętego tętna (thalach)')
    
    # macierz korelacji
    plt.subplot(2, 2, 4)
    corr = df.corr()
    sns.heatmap(corr[['target']].sort_values(by='target', ascending=False), 
                annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Korelacja cech ze zmienną docelową')
    
    plt.tight_layout()
    plt.show()

def preprocess_data(df):
    # ustawienie zero-jedynkowości dla targetu
    data = df.copy()
    data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    # dorzucenie mediany do brakujących wartości
    imputer = SimpleImputer(strategy='median')
    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return train_test_split(X_clean, y, test_size=0.2, random_state=42)