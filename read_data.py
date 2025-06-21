# read_data.py
from __future__ import annotations
"""Data loading utilities for Steam Game Analysis.

This module now loads **two** datasets:
1. The main CSV with numeric / date / platform info (`games.csv`).
2. A newline‑delimited JSON file with game descriptions and tags (`games_metadata.json`).

Both are returned as pandas DataFrames and pre‑processed for convenient use in the GUI & controller.
"""

"""
Moduł narzędzi do ładowania danych dla analizy gier Steam.

Obsługuje wczytywanie głównego pliku CSV oraz pliku JSON z metadanymi gier.
Zwraca przetworzone DataFrame'y gotowe do dalszej analizy i prezentacji.
"""

from pathlib import Path
import json
import logging
from typing import List

import pandas as pd

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths – ścieżki do plików z danymi i metadanymi
# ---------------------------------------------------------------------------
DATA_PATH: Path = config.DATA_PATH  # games.csv (already defined in config)
META_PATH: Path = config.BASE_DIR / "archive" / "games_metadata.json"  # JSON‑Lines

# ---------------------------------------------------------------------------
# Main CSV loader – wczytywanie i wstępne przetwarzanie głównego pliku z danymi
# ---------------------------------------------------------------------------

def load_data_game() -> pd.DataFrame:
    """
    Wczytuje i przetwarza główny zbiór danych *games.csv*.

    Kroki parsowania:
    - kolumny numeryczne konwertowane do float/int (błędy -> NaN)
    - `date_release` konwertowane do daty
    - flagi platform na bool
    - dodatkowa kolumna `title_len` z długością tytułu

    Zwraca:
        DataFrame z przetworzonymi danymi.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    # Wczytaj dane jako teksty (będzie konwersja poniżej)
    df = pd.read_csv(DATA_PATH, dtype=str, on_bad_lines="skip")

    # Konwersja kolumn numerycznych do float/int (błędne wartości -> NaN)
    numeric_cols: List[str] = [
        "app_id",
        "positive_ratio",
        "user_reviews",
        "price_final",
        "price_original",
        "discount",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Konwersja daty wydania do formatu datetime (tylko rok-miesiąc-dzień)
    if config.DATE_COL in df.columns:
        df[config.DATE_COL] = (
            df[config.DATE_COL]
            .astype(str)
            .str.split().str[0]
            .pipe(pd.to_datetime, errors="coerce")
        )

    # Flagi platform (np. Windows, Mac, Linux) na bool
    for col in config.PLATFORM_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"]).fillna(False)

    # Dodatkowa kolumna: długość tytułu gry
    if config.TITLE_COL in df.columns:
        df[config.TITLE_LEN_COL] = df[config.TITLE_COL].astype(str).str.len()

    return df

# ---------------------------------------------------------------------------
# Metadata loader (JSON‑Lines) – wczytywanie opisów i tagów gier z pliku JSON
# ---------------------------------------------------------------------------

def load_game_metadata() -> pd.DataFrame:
    """Load *games_metadata.json* containing description & tags.

    The file is newline‑delimited JSON (one object per line). Expected keys:
        * app_id : int
        * description : str
        * tags : list[str]

    Returns a DataFrame **indexed by `app_id`** for O(1) lookup.
    If the file is missing, an empty DF with the correct columns is returned.
    """
    if not META_PATH.exists():
        logger.warning("Metadata file not found: %s", META_PATH)
        # Zwraca pusty DataFrame z odpowiednimi kolumnami, jeśli plik nie istnieje
        empty = pd.DataFrame(columns=["description", "tags"])
        return empty.set_index(pd.Index([], name="app_id"))

    records = []
    # Wczytuje plik JSON‑Lines linia po linii
    with META_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError as e:
                logger.error("Bad JSON line in %s: %s", META_PATH, e)

    if not records:
        logger.warning("No valid metadata records found in %s", META_PATH)
        return pd.DataFrame(columns=["description", "tags"]).set_index("app_id")

    df_meta = pd.DataFrame.from_records(records)

    # Uzupełnia brakujące opisy pustym stringiem
    df_meta["description"] = df_meta["description"].fillna("")
    # Uzupełnia brakujące tagi pustą listą
    df_meta["tags"] = df_meta["tags"].apply(lambda x: x if isinstance(x, list) else [])

    # Upewnia się, że app_id jest liczbą całkowitą i ustawia jako indeks
    df_meta["app_id"] = pd.to_numeric(df_meta["app_id"], errors="coerce").astype("Int64")
    df_meta = df_meta.dropna(subset=["app_id"]).set_index("app_id")

    return df_meta

# ---------------------------------------------------------------------------
# Convenience combined loader (optional) – łączy dane główne z metadanymi
# ---------------------------------------------------------------------------

def load_games_with_metadata() -> pd.DataFrame:
    """Return the main games DataFrame **joined** with metadata by `app_id`."""
    # Wczytuje główny zbiór danych
    games = load_data_game()
    # Wczytuje metadane
    meta = load_game_metadata()
    # Jeśli nie ma metadanych, zwraca tylko główne dane
    if meta.empty:
        return games
    # Łączy dane główne z metadanymi po app_id (lewy join)
    return games.merge(meta, left_on="app_id", right_index=True, how="left")


if __name__ == "__main__":
    # Szybki test ładowania danych
    gdf = load_data_game()
    mdf = load_game_metadata()
    print("Loaded", len(gdf), "games; metadata for", len(mdf), "ids")
