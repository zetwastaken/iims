# dim_reduction.py
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

def wykonaj_pca(X, n_components=2):
    """
    Wykonuje redukcję wymiarów metodą PCA.
    Parametry: X - dane (np. macierz numpy), n_components - docelowa liczba składowych głównych.
    Zwraca krotkę: (X_zredukowane, czas_trwania).
    """
    start = time.time()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    czas_trwania = time.time() - start
    return X_pca, czas_trwania

def wykonaj_lda(X, y, n_components=2):
    """
    Wykonuje redukcję wymiarów metodą LDA (Linear Discriminant Analysis).
    Parametry: X - dane, y - etykiety klas, n_components - docelowa liczba wymiarów (maks. liczba klas-1).
    Zwraca krotkę: (X_zredukowane, czas_trwania).
    """
    # Uwaga: LDA może zredukować do maksymalnie (liczba_klas - 1) wymiarów
    n_classes = np.unique(y).size
    max_components = n_classes - 1
    if n_components > max_components:
        n_components = max_components  # dopasowanie liczby komponentów do ograniczenia LDA
    start = time.time()
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    czas_trwania = time.time() - start
    return X_lda, czas_trwania

def wykonaj_tsne(X, n_components=2, perplexity=30.0):
    """
    Wykonuje redukcję wymiarów metodą t-SNE.
    Parametry: X - dane, n_components - wymiar docelowy (najczęściej 2 lub 3), perplexity - parametr t-SNE.
    Zwraca krotkę: (X_zredukowane, czas_trwania).
    """
    start = time.time()
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=0, init='random')
    X_tsne = tsne.fit_transform(X)  # Uwaga: t-SNE jest nienadzorowane, y nie jest używane
    czas_trwania = time.time() - start
    return X_tsne, czas_trwania