# gui.py
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QComboBox, QSpinBox, QCheckBox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Importujemy funkcje z naszych modułów logiki
import data_manager
import dim_reduction
import plotting

class GlowneOkno(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Redukcja wymiarów - PCA/LDA/t-SNE")  # Ustaw tytuł okna
        self.resize(800, 600)  # Ustaw wstępny rozmiar okna

        # Inicjalizacja głównego widgetu centralnego i layoutu
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # *** Sekcja przycisków do wczytywania/generowania danych ***
        top_buttons_layout = QHBoxLayout()
        self.btnWczytaj = QPushButton("Wczytaj dane")      # przycisk do wczytywania danych z pliku
        self.btnGeneruj = QPushButton("Generuj dane")      # przycisk do generowania danych syntetycznych
        top_buttons_layout.addWidget(self.btnWczytaj)
        top_buttons_layout.addWidget(self.btnGeneruj)
        self.layout.addLayout(top_buttons_layout)

        # *** Sekcja wyboru metody redukcji i parametrów ***
        grid = QGridLayout()
        # Etykieta i pole wyboru metody
        grid.addWidget(QLabel("Metoda redukcji:"), 0, 0)
        self.comboMetoda = QComboBox()
        self.comboMetoda.addItems(["PCA", "LDA", "t-SNE"])
        grid.addWidget(self.comboMetoda, 0, 1)

        # Etykieta i pole liczby składowych (komponentów)
        grid.addWidget(QLabel("Liczba składowych:"), 1, 0)
        self.spinKomponenty = QSpinBox()
        self.spinKomponenty.setRange(1, 50)    # zakres możliwych wymiarów docelowych
        self.spinKomponenty.setValue(2)        # domyślnie redukcja do 2 wymiarów
        grid.addWidget(self.spinKomponenty, 1, 1)

        # Etykieta i pole parametru perplexity dla t-SNE
        grid.addWidget(QLabel("Perplexity:"), 2, 0)
        self.spinPerplexity = QSpinBox()
        self.spinPerplexity.setRange(1, 100)   # typowe wartości perplexity mieszczą się w tym zakresie
        self.spinPerplexity.setValue(30)       # domyślna wartość perplexity
        grid.addWidget(self.spinPerplexity, 2, 1)

        # Checkboxy do standaryzacji i normalizacji
        self.chkStandaryzacja = QCheckBox("Standaryzacja")   # (średnia=0, odchylenie=1)
        self.chkNormalizacja = QCheckBox("Normalizacja")     # (skalowanie do zakresu [0,1])
        grid.addWidget(self.chkStandaryzacja, 3, 0)
        grid.addWidget(self.chkNormalizacja, 3, 1)

        # Przyciski do uruchomienia redukcji i porównania metod
        self.btnRedukcja = QPushButton("Uruchom redukcję")
        self.btnPorownaj = QPushButton("Porównaj metody")
        grid.addWidget(self.btnRedukcja, 4, 0)
        grid.addWidget(self.btnPorownaj, 4, 1)

        # Dodajemy siatkę parametrów do głównego layoutu
        self.layout.addLayout(grid)

        # *** Sekcja wykresu (obszar Matplotlib) ***
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)    # Tworzymy obiekt Canvas Matplotlib
        self.layout.addWidget(self.canvas)         # Dodajemy Canvas jako widget do głównego layoutu

        # Inicjalizacja pól na dane
        self.X = None   # załadowane/wygenerowane dane cech (features)
        self.y = None   # załadowane/wygenerowane etykiety klas (labels), jeśli dostępne

        # Po utworzeniu elementów GUI, podłączamy sygnały do obsługi zdarzeń:
        self.btnWczytaj.clicked.connect(self.wczytaj_dane_z_pliku)
        self.btnGeneruj.clicked.connect(self.generuj_dane_syntetyczne)
        self.btnRedukcja.clicked.connect(self.uruchom_redukcje)
        self.btnPorownaj.clicked.connect(self.porownaj_metody)
        self.comboMetoda.currentTextChanged.connect(self.zmien_metode)

        # Ustawienie początkowe: pole perplexity aktywne tylko dla t-SNE
        self.zmien_metode(self.comboMetoda.currentText())

    def zmien_metode(self, metoda):
        """Slot wywoływany przy zmianie wybranej metody redukcji.
        Aktywuje/dezaktywuje pole 'Perplexity' dla metody t-SNE."""
        if metoda == "t-SNE":
            self.spinPerplexity.setEnabled(True)
        else:
            # Jeśli wybrano PCA lub LDA, pole perplexity nie ma zastosowania
            self.spinPerplexity.setEnabled(False)

    def wczytaj_dane_z_pliku(self):
        """Obsługa przycisku 'Wczytaj dane'. Otwiera okno dialogowe wyboru pliku i wczytuje dane CSV."""
        sciezka, _ = QFileDialog.getOpenFileName(self, "Wybierz plik danych CSV", "", "Pliki CSV (*.csv)")
        if sciezka:
            try:
                self.X, self.y = data_manager.wczytaj_dane_z_csv(sciezka)
                # Po pomyślnym wczytaniu możemy wyświetlić komunikat w status barze lub oknie dialogowym:
                liczba_próbek = self.X.shape[0] if self.X is not None else 0
                liczba_cech = self.X.shape[1] if self.X is not None else 0
                self.statusBar().showMessage(f"Wczytano dane: {liczba_próbek} próbek, {liczba_cech} cech.")
            except Exception as e:
                QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać pliku.\nSzczegóły: {e}")

    def generuj_dane_syntetyczne(self):
        """Obsługa przycisku 'Generuj dane'. Generuje przykładowe dane syntetyczne."""
        self.X, self.y = data_manager.generuj_dane_syntetyczne()
        # Komunikat o wygenerowaniu danych
        liczba_próbek = self.X.shape[0]
        liczba_cech = self.X.shape[1]
        self.statusBar().showMessage(f"Wygenerowano dane syntetyczne: {liczba_próbek} próbek, {liczba_cech} cech.")

    def uruchom_redukcje(self):
        """Obsługa przycisku 'Uruchom redukcję'. Wykonuje wybraną metodę redukcji wymiarów na aktualnych danych."""
        if self.X is None:
            QMessageBox.warning(self, "Brak danych", "Najpierw wczytaj lub wygeneruj dane do analizy.")
            return

        # Pobranie ustawień od użytkownika
        metoda = self.comboMetoda.currentText()
        n_components = self.spinKomponenty.value()
        perplexity = self.spinPerplexity.value()

        # Opcjonalna standaryzacja/normalizacja danych
        X_proc = self.X  # X po przetworzeniu
        if self.chkStandaryzacja.isChecked():
            X_proc = data_manager.standaryzuj_dane(X_proc)
        if self.chkNormalizacja.isChecked():
            X_proc = data_manager.normalizuj_dane(X_proc)

        # Wykonanie wybranej metody redukcji
        czas_trwania = 0
        if metoda == "PCA":
            X_red, czas_trwania = dim_reduction.wykonaj_pca(X_proc, n_components=n_components)
        elif metoda == "LDA":
            if self.y is None:
                QMessageBox.warning(self, "Brak etykiet", "Metoda LDA wymaga danych z etykietami klas.")
                return
            X_red, czas_trwania = dim_reduction.wykonaj_lda(X_proc, self.y, n_components=n_components)
        elif metoda == "t-SNE":
            # Uwaga: t-SNE jest metodą nienadzorowaną, etykiety nie są wykorzystywane
            X_red, czas_trwania = dim_reduction.wykonaj_tsne(X_proc, n_components=n_components, perplexity=perplexity)
        else:
            QMessageBox.warning(self, "Nieznana metoda", f"Nieobsługiwana metoda: {metoda}")
            return

        # Rysowanie wyniku na wykresie
        ax = self.figure.add_subplot(111, projection='3d' if X_red.shape[1] == 3 else None)
        ax.clear()  # wyczyść ewentualne stare dane na osi
        if X_red.shape[1] == 3:
            plotting.rysuj_wykres_3d(ax, X_red, self.y)
        else:
            plotting.rysuj_wykres_2d(ax, X_red, self.y)
        ax.set_title(f"Wynik {metoda}")  # tytuł wykresu z nazwą metody
        self.canvas.draw()  # odświeżenie canvasu, aby pokazać nowy wykres

        # Wyświetlenie czasu wykonania w pasku statusu
        self.statusBar().showMessage(f"Metoda {metoda} zakończona w {czas_trwania:.4f} s")

    def porownaj_metody(self):
        """Obsługa przycisku 'Porównaj metody'. Wykonuje PCA, LDA i t-SNE kolejno i wyświetla porównanie."""
        if self.X is None:
            QMessageBox.warning(self, "Brak danych", "Najpierw wczytaj lub wygeneruj dane do analizy.")
            return

        # Przygotowanie danych (opcjonalna standaryzacja/normalizacja)
        X_proc = self.X
        if self.chkStandaryzacja.isChecked():
            X_proc = data_manager.standaryzuj_dane(X_proc)
        if self.chkNormalizacja.isChecked():
            X_proc = data_manager.normalizuj_dane(X_proc)

        # Dla porównania ustawiamy redukcję do 2 wymiarów (dla czytelności wykresów 2D obok siebie)
        n_components = 2  
        perplexity = self.spinPerplexity.value()

        # Wykonanie metod (LDA wymaga etykiet)
        X_pca, t_pca = dim_reduction.wykonaj_pca(X_proc, n_components=n_components)
        X_lda, t_lda = None, 0
        if self.y is not None:
            X_lda, t_lda = dim_reduction.wykonaj_lda(X_proc, self.y, n_components=n_components)
        X_tsne, t_tsne = dim_reduction.wykonaj_tsne(X_proc, n_components=n_components, perplexity=perplexity)

        # Przygotowanie wykresu porównawczego z trzema sub-plotami
        self.figure.clear()
        fig = self.figure
        # Tworzymy 3 osi obok siebie
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        # Rysujemy wyniki na każdej z osi
        plotting.rysuj_wykres_2d(ax1, X_pca, self.y)
        ax1.set_title(f"PCA ({t_pca:.3f}s)")
        if X_lda is not None:
            plotting.rysuj_wykres_2d(ax2, X_lda, self.y)
            ax2.set_title(f"LDA ({t_lda:.3f}s)")
        else:
            ax2.text(0.5, 0.5, "LDA: brak etykiet", ha='center', va='center')
            ax2.set_title("LDA")
        plotting.rysuj_wykres_2d(ax3, X_tsne, self.y)
        ax3.set_title(f"t-SNE ({t_tsne:.3f}s)")
        fig.tight_layout()  # opcjonalnie dopasuj odstępy między wykresami
        self.canvas.draw()

        # Komunikat w pasku statusu z czasami (zbiorczo)
        self.statusBar().showMessage(f"PCA: {t_pca:.4f}s, LDA: {t_lda:.4f}s, t-SNE: {t_tsne:.4f}s")