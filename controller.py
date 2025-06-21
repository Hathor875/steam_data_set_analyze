# controller.py
"""
Moduł kontrolera – walidacja, łączenie metadanych i wybór strategii analizy.

Obsługuje walidację dat i cen, pobieranie metadanych oraz uruchamianie wybranych strategii analitycznych.
"""

from __future__ import annotations
import importlib
import logging
import re
from typing import Optional, Union, Dict

import pandas as pd
from pandas import DataFrame

import config
from read_data import load_game_metadata  # new metadata loader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ładowanie metadanych przy imporcie – wczytuje metadane do DataFrame indeksowanego po app_id
# ---------------------------------------------------------------------------
_meta_df: DataFrame = load_game_metadata()  # indeksowane po app_id
logger.info("Załadowano metadane dla %d app_ids", len(_meta_df))

# ---------------------------------------------------------------------------
# Pomocnicze funkcje konwersji – zamiana tekstu na datę lub liczbę, walidacja wartości
# ---------------------------------------------------------------------------

def _coerce_date(s: str | None) -> Optional[pd.Timestamp]:
    """
    Konwertuje napis na obiekt daty Pandas lub zwraca None.
    """
    return pd.to_datetime(s) if s else None


def _coerce_price(v) -> Optional[float]:
    """
    Konwertuje wartość na liczbę zmiennoprzecinkową (cena) lub zwraca None.
    Zgłasza wyjątek, jeśli cena jest ujemna lub nieprawidłowa.
    """
    if v in (None, ""):
        return None
    try:
        num = float(v)
        if num < 0:
            raise ValueError
        return num
    except (TypeError, ValueError):
        raise ValueError("Cena musi być nieujemną liczbą.")

# ---------------------------------------------------------------------------
# Walidacja dat i cen – sprawdzanie poprawności zakresów podanych przez użytkownika
# ---------------------------------------------------------------------------

def validate_dates(df: DataFrame, start: str | None, end: str | None) -> Optional[str]:
    """
    Sprawdza poprawność zakresu dat. Zwraca komunikat o błędzie lub None.
    """
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if start and not date_pattern.match(start):
        return "Format daty musi być RRRR-MM-DD"
    if end and not date_pattern.match(end):
        return "Format daty musi być RRRR-MM-DD"
    mn, mx = df[config.DATE_COL].min(), df[config.DATE_COL].max()
    try:
        if start and (_coerce_date(start) < mn):
            return f"Data początkowa nie może być wcześniejsza niż {mn.date()}"
        if end and (_coerce_date(end) > mx):
            return f"Data końcowa nie może być późniejsza niż {mx.date()}"
    except Exception:
        return "Format daty musi być RRRR-MM-DD"
    return None


def _validate_prices(pmin_raw, pmax_raw) -> tuple[Optional[float], Optional[float]]:
    """
    Sprawdza poprawność zakresu cen. Zwraca krotkę (min, max) lub zgłasza wyjątek.
    """
    pmin, pmax = _coerce_price(pmin_raw), _coerce_price(pmax_raw)
    if pmin is not None and pmax is not None and pmin > pmax:
        raise ValueError("Cena minimalna nie może być większa od maksymalnej.")
    return pmin, pmax

# ---------------------------------------------------------------------------
# Wybór i uruchamianie strategii analizy – dynamiczne tworzenie i uruchamianie klasy analitycznej
# ---------------------------------------------------------------------------

def run_analysis(
    df: DataFrame,
    analysis_key: str,
    platforms: list[str],
    start: str | None,
    end: str | None,
    price_min=None,
    price_max=None,
    **extra,
):
    """
    Waliduje dane wejściowe, tworzy strategię analizy, uruchamia ją i zwraca (figury, wnioski).
    """
    # Walidacja cen
    pmin, pmax = _validate_prices(price_min, price_max)
    # Pobranie klasy strategii na podstawie klucza
    class_name = config.ANALYSIS_MAP.get(analysis_key)
    if not class_name:
        raise ValueError(f"Brak strategii dla analizy: {analysis_key}")
    Strategy = getattr(importlib.import_module("analyze"), class_name)
    # Utworzenie instancji strategii z przekazanymi parametrami
    strat = Strategy(
        platforms=platforms,
        start_date=start,
        end_date=end,
        price_min=pmin,
        price_max=pmax,
        **extra,
    )
    # Uruchomienie analizy i pobranie wyników
    figs = strat.execute(strat.preprocess(df))
    return figs, strat.get_insights()

# ---------------------------------------------------------------------------
# Pomocnicze funkcje wyszukiwania i łączenia metadanych – pobieranie i łączenie danych o grach
# ---------------------------------------------------------------------------

def _merge_meta(record: Dict) -> Dict:
    """
    Jeśli istnieją metadane dla `app_id`, dołącza opis i tagi do rekordu.
    """
    app_id = record.get("app_id")
    if app_id is not None and app_id in _meta_df.index:
        meta = _meta_df.loc[app_id].to_dict()
        record.update(meta)
    return record


def _row(df: DataFrame, col: str, val) -> Optional[dict]:
    """
    Zwraca pełny rekord (dane + metadane) dla pierwszego wiersza, gdzie col == val.
    """
    try:
        num = int(val)
    except (ValueError, TypeError):
        return None
    row = df.loc[df[col] == num]
    if row.empty:
        return None
    rec = row.iloc[0].to_dict()
    # Jeśli rekord jest pusty (np. po przekroczeniu zakresu indeksu), zwróć None
    if not rec or all(v is None for v in rec.values()):
        return None
    return _merge_meta(rec)


def get_game_info_by_id(df: DataFrame, gid) -> Optional[dict]:
    """
    Zwraca informacje o grze na podstawie app_id.
    """
    return _row(df, "app_id", gid)


def get_game_info_by_row(df: DataFrame, idx) -> Optional[dict]:
    """
    Zwraca informacje o grze na podstawie numeru wiersza.
    """
    try:
        i = int(idx)
    except (ValueError, TypeError):
        return None
    if i < 0 or i >= len(df):
        return None
    rec = df.iloc[i].to_dict()
    if not rec or all(v is None for v in rec.values()):
        return None
    return _merge_meta(rec)
