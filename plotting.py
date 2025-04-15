# plotting.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # wymagane do projekcji 3D (może być nieużywane bez 3D)

def rysuj_wykres_2d(ax, X, y=None):
    """
    Rysuje wykres punktowy 2D na podanej osi matplotlib.
    X - macierz (n_próbek, 2) ze zredukowanymi współrzędnymi 2D.
    y - wektor etykiet klas (opcjonalny). Jeśli podany, punkty zostaną pokolorowane wg klas.
    """
    ax.clear()  # wyczyść oś przed rysowaniem (dla pewności)
    if y is not None:
        y = np.array(y)
        unique_classes = np.unique(y)
        colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
        # Jeżeli klas jest więcej niż zdefiniowanych kolorów, można rozszerzyć listę lub użyć pętli modulo.
        for idx, cls in enumerate(unique_classes):
            kolor = colors[idx % len(colors)]
            # Wybieramy punkty należące do klasy cls
            X_cls = X[y == cls]
            ax.scatter(X_cls[:, 0], X_cls[:, 1], c=kolor, label=str(cls), alpha=0.8)
        ax.legend(title="Klasa")
    else:
        # Brak etykiet - rysujemy wszystkie punkty jednym kolorem
        ax.scatter(X[:, 0], X[:, 1], c="blue", alpha=0.8)
    ax.set_xlabel("Składowa 1")
    ax.set_ylabel("Składowa 2")

def rysuj_wykres_3d(ax, X, y=None):
    """
    Rysuje wykres punktowy 3D na podanej osi (ax) matplotlib.
    X - macierz (n_próbek, 3) ze zredukowanymi współrzędnymi 3D.
    y - wektor etykiet klas (opcjonalny) do kolorowania punktów.
    """
    ax.clear()
    if y is not None:
        y = np.array(y)
        unique_classes = np.unique(y)
        colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
        for idx, cls in enumerate(unique_classes):
            kolor = colors[idx % len(colors)]
            X_cls = X[y == cls]
            ax.scatter(X_cls[:, 0], X_cls[:, 1], X_cls[:, 2], c=kolor, label=str(cls), alpha=0.8)
        ax.legend(title="Klasa")
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c="blue", alpha=0.8)
    ax.set_xlabel("Składowa 1")
    ax.set_ylabel("Składowa 2")
    ax.set_zlabel("Składowa 3")