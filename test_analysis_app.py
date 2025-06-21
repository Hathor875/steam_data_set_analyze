"""
Testy jednostkowe dla aplikacji analizy gier Steam.

Ten plik zawiera testy sprawdzające poprawność działania kluczowych elementów aplikacji:
- Walidacja danych wejściowych: testy sprawdzają poprawność obsługi zakresów dat i cen, w tym przypadki brzegowe i błędne formaty.
- Funkcje pobierania informacji o grach: testy weryfikują, czy można poprawnie znaleźć grę po ID lub numerze wiersza oraz czy obsługiwane są przypadki braku danych.

Testy korzystają z przykładowego DataFrame (fixture sample_df) i pokrywają zarówno przypadki pozytywne, jak i negatywne.

Aby uruchomić testy, użyj polecenia:
    pytest test_analysis_app.py

"""
import pytest
import pandas as pd
import numpy as np
from analyze import *
from controller import validate_dates, run_analysis, get_game_info_by_id, get_game_info_by_row
import config

# ---------------- Fixtures and Helpers ------------------

@pytest.fixture
def sample_df():
    """
    Przykładowy DataFrame do testów.
    """
    data = {
        "app_id": [1, 2, 3],
        "title": ["Game A", "Game B", "Game C"],
        "positive_ratio": [80, 70, 90],
        "user_reviews": [1000, 1500, 500],
        "price_final": [10.0, 20.0, 15.0],
        "price_original": [20.0, 25.0, 20.0],
        "discount": [0.5, 0.2, 0.25],
        "date_release": pd.to_datetime(["2020-01-01", "2021-06-15", "2022-12-31"]),
        "win": [True, False, True],
        "mac": [False, True, False],
        "linux": [True, True, False],
        "steam_deck": [False, False, True],
        "tags": [["indie", "rpg"], ["action"], ["puzzle"]]
    }
    return pd.DataFrame(data)


# ---------------- Unit Tests ------------------

def test_platform_count_strategy(sample_df):
    # Testuje strategię liczenia gier na platformę: czy działa filtracja, generowanie wykresu i wnioski
    strat = PlatformCountStrategy(platforms=["win", "mac"])
    df_f = strat.preprocess(sample_df)
    fig = strat.execute(df_f)
    assert fig is not None
    assert "win" in df_f.columns
    assert strat.get_insights().startswith("Najwięcej gier")

def test_platform_rating_strategy(sample_df):
    # Testuje strategię rozkładu ocen na platformach
    strat = PlatformRatingsStrategy(platforms=["win", "linux"])
    df_f = strat.preprocess(sample_df)
    fig = strat.execute(df_f)
    assert fig is not None

def test_rating_review_strategy(sample_df):
    # Testuje strategię zależności ocen od liczby recenzji
    strat = RatingReviewStrategy()
    df_f = strat.preprocess(sample_df)
    fig = strat.execute(df_f)
    assert fig is not None
    insight = strat.get_insights()
    assert "ρ" in insight

def test_trend_over_time_strategy(sample_df):
    # Testuje strategię trendów wydawania gier i cen w czasie
    strat = TrendOverTimeStrategy()
    df_f = strat.preprocess(sample_df)
    figs = strat.execute(df_f)
    assert len(figs) == 2

def test_release_day_strategy(sample_df):
    # Testuje strategię rozkładu premier po dniach tygodnia
    strat = ReleaseDayStrategy()
    df_f = strat.preprocess(sample_df)
    fig = strat.execute(df_f)
    assert fig is not None

def test_games_by_month_strategy(sample_df):
    # Testuje strategię rozkładu premier po miesiącach
    strat = GamesByMonthStrategy()
    df_f = strat.preprocess(sample_df)
    fig = strat.execute(df_f)
    assert fig is not None

def test_advanced_multi_param_strategy(sample_df):
    # Testuje strategię zależności ceny, oceny i liczby recenzji
    strat = AdvancedMultiParamStrategy()
    df_f = strat.preprocess(sample_df)
    fig = strat.execute(df_f)
    assert fig is not None

def test_3d_scatter_strategy(sample_df):
    # Testuje strategię 3D: cena, recenzje, ocena
    strat = ThreeDScatterStrategy()
    df_f = strat.preprocess(sample_df)
    fig = strat.execute(df_f)
    assert fig is not None

def test_custom_scatter_strategy(sample_df):
    # Testuje strategię własnego wykresu punktowego
    strat = CustomScatterStrategy(x_col="price_final", y_col="positive_ratio")
    df_f = strat.preprocess(sample_df)
    fig = strat.execute(df_f)
    assert fig is not None

def test_top_tags_strategy(sample_df):
    # Testuje strategię najpopularniejszych tagów
    strat = TopTagsStrategy()
    df_f = strat.preprocess(sample_df)
    fig = strat.execute(df_f)
    assert fig is not None

def test_validate_dates_correct(sample_df):
    # Sprawdza, czy walidacja dat akceptuje poprawny zakres
    start = "2020-01-01"
    end = "2022-12-31"
    err = validate_dates(sample_df, start, end)
    assert err is None

def test_validate_dates_out_of_range(sample_df):
    # Sprawdza, czy walidacja dat wykrywa datę spoza zakresu
    start = "2010-01-01"
    err = validate_dates(sample_df, start, None)
    assert "nie może być wcześniejsza" in err

def test_validate_dates_invalid_format(sample_df):
    # Sprawdza, czy walidacja dat wykrywa niepoprawny format daty
    start = "2020/01/01"  # niepoprawny format
    err = validate_dates(sample_df, start, None)
    assert "Format daty" in err

def test_validate_dates_end_too_late(sample_df):
    # Sprawdza, czy walidacja dat wykrywa zbyt późną datę końcową
    end = "2030-01-01"
    err = validate_dates(sample_df, None, end)
    assert "nie może być późniejsza" in err

def test_validate_dates_ok_none(sample_df):
    # Sprawdza, czy walidacja dat akceptuje brak dat
    err = validate_dates(sample_df, None, None)
    assert err is None

def test_validate_prices_valid():
    # Sprawdza poprawność walidacji cen dla poprawnych danych
    from controller import _validate_prices
    assert _validate_prices(10, 20) == (10, 20)
    assert _validate_prices(None, 20) == (None, 20)
    assert _validate_prices(10, None) == (10, None)
    assert _validate_prices(None, None) == (None, None)

def test_validate_prices_invalid():
    # Sprawdza, czy walidacja cen wykrywa błędne dane i zgłasza wyjątek
    from controller import _validate_prices
    import pytest
    with pytest.raises(ValueError):
        _validate_prices(20, 10)
    with pytest.raises(ValueError):
        _validate_prices(-1, 10)
    with pytest.raises(ValueError):
        _validate_prices(10, -5)

def test_run_analysis_valid(sample_df):
    # Testuje uruchomienie analizy na poprawnych danych
    figs, insight = run_analysis(
        df=sample_df,
        analysis_key="Rating vs Reviews",
        platforms=["win", "mac"],
        start="2020-01-01",
        end="2022-12-31"
    )
    assert insight is not None

def test_get_game_info_by_id_found(sample_df):
    # Sprawdza, czy funkcja poprawnie znajduje grę po app_id
    record = get_game_info_by_id(sample_df, 1)
    assert record is not None
    assert record["title"] == "Game A"

def test_get_game_info_by_id_not_found(sample_df):
    # Sprawdza, czy funkcja zwraca None, gdy nie ma gry o podanym app_id
    record = get_game_info_by_id(sample_df, 999)
    assert record is None

def test_get_game_info_by_row_found(sample_df):
    # Sprawdza, czy funkcja poprawnie znajduje grę po numerze wiersza
    record = get_game_info_by_row(sample_df, 0)
    assert record is not None
    assert record["title"] == "Game A"

def test_get_game_info_by_row_not_found(sample_df):
    # Sprawdza, czy funkcja zwraca None, gdy nie ma gry o podanym numerze wiersza
    record = get_game_info_by_row(sample_df, 999)
    assert record is None
