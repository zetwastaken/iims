# data_manager.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def wczytaj_dane_z_csv(sciezka_pliku):
    """
    Wczytuje dane z pliku CSV o podanej ścieżce.
    Zakłada, że ostatnia kolumna zawiera etykiety klas (jeśli dostępne).
    Zwraca krotkę (X, y): X - numpy array z danymi cech, y - wektor etykiet (lub None jeśli brak).
    """
    df = pd.read_csv(sciezka_pliku)
    if df.shape[1] < 2:
        # Jeśli jest tylko jedna kolumna, traktujemy to jako dane bez etykiet
        X = df.values
        y = None
    else:
        # Przyjmujemy, że ostatnia kolumna to etykieta klasy (kategorii)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        # Jeśli etykiety są typu nie-numerycznego (np. tekstowe), konwertujemy je na kody liczbowe
        if y.dtype == object or y.dtype == str:
            y = pd.Categorical(y).codes
    return X, y

def generuj_dane_syntetyczne(n_samples=300, n_features=5, n_classes=3, random_state=0):
    """
    Generuje losowe dane syntetyczne do testów.
    Domyślnie tworzy n_samples punktów o n_features cechach, należących do n_classes klas.
    Zwraca krotkę (X, y).
    """
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, random_state=random_state)
    return X, y

def standaryzuj_dane(X):
    """
    Przeprowadza standaryzację cech: przeskalowanie do średniej 0 i odchylenia standardowego 1.
    Zwraca przetransformowaną macierz X.
    """
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std

def normalizuj_dane(X):
    """
    Przeprowadza normalizację cech: skalowanie do przedziału [0,1].
    (Min-Max scaling dla każdej cechy niezależnie).
    Zwraca przetransformowaną macierz X.
    """
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm