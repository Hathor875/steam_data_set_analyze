# Steam Game Analysis

## Project Description

The application allows you to load, explore, and analyze data about games from the Steam store. The user can filter data, select date ranges, platforms, prices, and generate various analyses and visualizations. Results are presented as charts and textual insights.

## Justification of Methods Used

- **Analyses and visualizations** are tailored to the nature of the data (e.g., rating distribution, number of reviews, prices, tag popularity), allowing for quick identification of trends and relationships.
- **Filtering and type conversions** (e.g., dates, numbers, categories) enable flexible data exploration.
- **Visualizations** (bar, line, box, and 3D charts) are clear and suited to the analyzed features.

## Programming Solutions

- **Object-oriented programming**: Key elements (analysis strategies, GUI, controller) are implemented as classes.
- **Strategy design pattern**: Allows easy addition of new analysis types without modifying the main program code.
- **Unit tests**: The `test_analysis_app.py` file checks the correctness of analytical functions and the controller.
- **Exception handling**: The program handles user errors and data loading errors, ensuring uninterrupted execution.

## Project Structure

- `main.py` – application entry point
- `gui.py` – graphical user interface
- `controller.py` – control logic, validation, strategy selection
- `analyze.py` – data analysis strategies (Strategy pattern)
- `read_data.py` – data loading and preprocessing
- `config.py` – constants and settings
- `test_analysis_app.py` – unit tests
- `archive/games.csv`, `archive/games_metadata.json` – data
- `requirements.txt` – required libraries

## Requirements

- Python 3.10+
- All required libraries are listed in `requirements.txt`

## Running the Application

1. Install required libraries:  
   `pip install -r requirements.txt`
2. Run the application:  
   `python main.py`

## Author

Krzysztof Cieślik

---

# Opis projektu (Polska wersja)

## Opis projektu

Aplikacja umożliwia wczytywanie, eksplorację i analizę danych o grach ze sklepu Steam. Użytkownik może filtrować dane, wybierać zakresy dat, platformy, ceny oraz generować różne analizy i wizualizacje. Wyniki prezentowane są w formie wykresów oraz tekstowych wniosków.

## Uzasadnienie zastosowanych metod

- **Analizy i wizualizacje** zostały dobrane do charakteru danych (np. rozkład ocen, liczba recenzji, ceny, popularność tagów), co pozwala na szybkie wychwycenie trendów i zależności.
- **Filtrowanie i konwersje typów** (np. daty, liczby, kategorie) umożliwiają elastyczną eksplorację zbioru danych.
- **Wizualizacje** (wykresy słupkowe, liniowe, pudełkowe, 3D) są czytelne i dostosowane do analizowanych cech.

## Rozwiązania programistyczne

- **Programowanie obiektowe**: Kluczowe elementy (strategie analizy, GUI, kontroler) są zaimplementowane jako klasy.
- **Wzorzec projektowy Strategia**: Umożliwia łatwe dodawanie nowych typów analiz bez modyfikacji kodu głównego programu.
- **Testy jednostkowe**: Plik `test_analysis_app.py` sprawdza poprawność działania funkcji analitycznych i kontrolera.
- **Obsługa wyjątków**: Program obsługuje błędy użytkownika i błędy podczas ładowania danych, zapewniając nieprzerwaną egzekucję.

## Struktura projektu

- `main.py` – uruchomienie aplikacji
- `gui.py` – interfejs graficzny
- `controller.py` – logika sterująca, walidacja, wybór strategii
- `analyze.py` – strategie analizy danych (wzorzec Strategia)
- `read_data.py` – ładowanie i wstępne przetwarzanie danych
- `config.py` – stałe i ustawienia
- `test_analysis_app.py` – testy jednostkowe
- `archive/games.csv`, `archive/games_metadata.json` – dane
- `requirements.txt` – wymagane biblioteki

## Wymagania

- Python 3.10+
- Wszystkie wymagane biblioteki znajdują się w `requirements.txt`

## Uruchomienie

1. Zainstaluj wymagane biblioteki:  
   `pip install -r requirements.txt`
2. Uruchom aplikację:  
   `python main.py`

## Autor

Krzysztof Cieślik
