# main.py
import sys
from PyQt5.QtWidgets import QApplication
import gui  # import modułu gui zawierającego definicję głównego okna

def main():
    """Główny punkt wejścia aplikacji."""
    app = QApplication(sys.argv)          # Tworzenie obiektu aplikacji
    okno = gui.GlowneOkno()              # Inicjalizacja głównego okna GUI
    okno.show()                          # Wyświetlenie okna na ekranie
    sys.exit(app.exec_())                # Uruchomienie pętli zdarzeń aplikacji

if __name__ == "__main__":
    main()