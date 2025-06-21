"""
Moduł konfiguracyjny - stałe i ustawienia dla aplikacji analizy gier Steam.

Zawiera ścieżki do plików, nazwy kolumn, mapowanie analiz oraz teksty interfejsu.
"""

# config.py
from pathlib import Path

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "archive" / "games.csv"

# --------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------
WINDOW_TITLE = "Steam Game Analysis"
WINDOW_SIZE  = "1280x800"

# --------------------------------------------------------------------
# Data columns
# --------------------------------------------------------------------
PLATFORM_COLS = ["win", "mac", "linux", "steam_deck"]
RATING_COL    = "positive_ratio"
REVIEW_COL    = "user_reviews"
VALUE_COL     = "price_final"
DATE_COL      = "date_release"
TITLE_COL     = "title"
TITLE_LEN_COL = "title_len"          # derived in read_data.py

# --------------------------------------------------------------------
# Analysis mapping
# --------------------------------------------------------------------
ANALYSIS_MAP = {
    "Game count by platforms"         : "PlatformCountStrategy",
    "Rating vs Reviews"               : "RatingReviewStrategy",
    "Trends over time"                : "TrendOverTimeStrategy",
    "Rating distribution by platform" : "PlatformRatingsStrategy",
    "Games by day of the week"        : "ReleaseDayStrategy",
    "Games by month"                  : "GamesByMonthStrategy",
    "Advanced multi-param analysis"   : "AdvancedMultiParamStrategy",
    "3D Price-Reviews-Rating"         : "ThreeDScatterStrategy",
    "Custom 2D scatter"               : "CustomScatterStrategy",
    "Top tags overall"                : "TopTagsStrategy",
    "Avg price by tag"    : "AvgPriceByTagStrategy",
    "Tag co-occurrence"   : "TagCoOccurrenceStrategy",
    "Reviews over time"   : "ReviewsOverTimeStrategy",
}


# --------------------------------------------------------------------
# UI text
# --------------------------------------------------------------------
ABOUT_TEXT        = "Steam Game Analysis App\nCreated for data exploration and visualization."
GET_BY_ID_LABEL   = "Get game by ID"
GET_BY_ROW_LABEL  = "Get row number"
GET_BY_ID_TITLE   = "Game Info - ID {id}"
GET_BY_ROW_TITLE  = "Game Info - Row {row}"
INVALID_INPUT_MSG = "Input must be a positive number."
NOT_FOUND_MSG     = "No matching entry found."
