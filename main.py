"""
Główny plik uruchamiający aplikację GUI do analizy gier Steam.

Uruchamia okno aplikacji i obsługuje sygnał przerwania (Ctrl+C).
"""

import signal
from gui import App

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    App().mainloop()
