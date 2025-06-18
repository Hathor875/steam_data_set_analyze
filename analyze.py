"""
Strategie analizy – filtr platformy + wnioski + analiza miesięczna.

Ten moduł wykorzystuje wzorzec projektowy Strategia (Strategy), aby zapewnić elastyczny sposób wykonywania różnych typów analiz danych o grach. Abstrakcyjna klasa bazowa `AnalysisStrategy` definiuje wspólny interfejs dla wszystkich strategii analizy. Konkretne klasy strategii (np. PlatformCountStrategy, PlatformRatingsStrategy itd.) implementują specyficzne zachowania analityczne poprzez nadpisanie metod `execute` i `get_insights`. Pozwala to na wybór i zmianę logiki analizy w czasie działania programu bez modyfikowania kodu klienta, co sprzyja modularności i skalowalności.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import pearsonr

import config

warnings.filterwarnings(
    "ignore", category=UserWarning, module=r"matplotlib\.category"
)
sns.set_style("whitegrid")


class AnalysisStrategy(ABC):
    """
    Abstrakcyjna klasa bazowa dla strategii analizy.

    Definiuje interfejs do wstępnego przetwarzania danych, wykonania analizy
    oraz pobierania wniosków. Konkretne strategie powinny dziedziczyć po tej klasie.
    """

    def __init__(
        self,
        platforms: List[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
        **_,
    ) -> None:
        """
        Inicjalizuje strategię analizy z opcjonalnymi filtrami.

        Argumenty:
            platforms: Lista kolumn platform do uwzględnienia.
            start_date: Filtr gier wydanych po tej dacie.
            end_date: Filtr gier wydanych przed tą datą.
            price_min: Minimalna cena.
            price_max: Maksymalna cena.
        """
        self.platforms = platforms or config.PLATFORM_COLS
        self.start = pd.to_datetime(start_date) if start_date else None
        self.end = pd.to_datetime(end_date) if end_date else None
        self.price_min = price_min
        self.price_max = price_max
        self._df: pd.DataFrame | None = None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zastosuj filtry do DataFrame na podstawie parametrów inicjalizacji.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Przefiltrowany DataFrame.
        """
        df_f = df.copy()

        if self.start is not None:
            df_f = df_f[df_f[config.DATE_COL] >= self.start]
        if self.end is not None:
            df_f = df_f[df_f[config.DATE_COL] <= self.end]
        if self.price_min is not None:
            df_f = df_f[df_f[config.VALUE_COL] >= self.price_min]
        if self.price_max is not None:
            df_f = df_f[df_f[config.VALUE_COL] <= self.price_max]

        valid = [p for p in self.platforms if p in df_f.columns]
        if not valid:
            raise ValueError("Wybrane kolumny platform nie istnieją w zbiorze danych.")
        df_f = df_f[df_f[valid].any(axis=1)]

        self._df = df_f
        return df_f

    @abstractmethod
    def execute(self, df: pd.DataFrame) -> Union[Figure, Tuple[Figure, ...]]:
        """
        Wykonaj analizę na przekazanym DataFrame.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure lub krotkę Figure z wynikami analizy.
        """
        ...

    def get_insights(self) -> str:
        """
        Zwraca tekst z wnioskami lub podsumowaniem analizy.

        Zwraca:
            Wnioski jako tekst.
        """
        return ""


class PlatformCountStrategy(AnalysisStrategy):
    """
    Strategia liczenia liczby gier na platformę.
    """

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres słupkowy liczby gier na platformę.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        counts = df[self.platforms].sum()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set(title="Liczba gier na platformę",
               xlabel="Platforma", ylabel="Liczba gier")
        return fig

    def get_insights(self) -> str:
        """
        Zwraca informację o platformie z największą i najmniejszą liczbą gier.

        Zwraca:
            Wnioski jako tekst.
        """
        counts = self._df[self.platforms].sum()
        return (f"Najwięcej gier ukazało się na platformie "
                f"{counts.idxmax().capitalize()}, "
                f"najmniej na {counts.idxmin().capitalize()}.")


class PlatformRatingsStrategy(AnalysisStrategy):
    """
    Strategia analizy rozkładu ocen na platformach.
    """

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres pudełkowy ocen dla każdej platformy.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        long_df = pd.concat(
            [
                pd.DataFrame(
                    {"Platform": p.capitalize(),
                     "Rating": df.loc[df[p], config.RATING_COL].dropna()}
                )
                for p in self.platforms if p in df.columns
            ],
            ignore_index=True,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=long_df, x="Platform", y="Rating", ax=ax, orientation="vertical")
        ax.set(title="Rozkład ocen na platformach", ylabel="Pozytywna ocena (%)")
        return fig

    def get_insights(self) -> str:
        """
        Zwraca informację o platformie z najwyższą medianą ocen.

        Zwraca:
            Wnioski jako tekst.
        """
        med = {
            p.capitalize(): self._df.loc[self._df[p], config.RATING_COL].median()
            for p in self.platforms if p in self._df.columns
        }
        best = max(med, key=med.get)
        return f"Platforma {best} ma najwyższą medianę ocen (~{med[best]:.1f} %)."


class RatingReviewStrategy(AnalysisStrategy):
    """
    Strategia analizy zależności między ocenami a liczbą recenzji.
    """

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres punktowy ocen względem liczby recenzji.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=config.REVIEW_COL, y=config.RATING_COL,
                        ax=ax, alpha=0.6)
        ax.set_xscale("log")
        ax.set(title="Ocena względem liczby recenzji",
               xlabel="Recenzje użytkowników (skala log)",
               ylabel="Pozytywna ocena (%)")
        return fig

    def get_insights(self) -> str:
        """
        Zwraca korelację między recenzjami a ocenami.

        Zwraca:
            Wnioski jako tekst.
        """
        d = self._df.dropna(subset=[config.REVIEW_COL, config.RATING_COL])
        corr, _ = pearsonr(np.log10(d[config.REVIEW_COL] + 1), d[config.RATING_COL])
        return f"Korelacja recenzje–ocena wynosi ρ ≈ {corr:.2f}."


class TrendOverTimeStrategy(AnalysisStrategy):
    """
    Strategia analizy trendów wydawania gier i cen w czasie.
    """

    def execute(self, df: pd.DataFrame) -> Tuple[Figure, Figure]:
        """
        Rysuje liczbę wydanych gier na rok oraz średnią cenę na rok.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Krotka Figure.
        """
        df = df.dropna(subset=[config.DATE_COL]).copy()
        df["year"] = df[config.DATE_COL].dt.year
        self._df = df

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.countplot(x="year", data=df, ax=ax1)
        ax1.set(title="Liczba gier wydanych na rok")

        yearly = df.groupby("year")[config.VALUE_COL].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=yearly, x="year", y=config.VALUE_COL,
                     marker="o", ax=ax2)
        ax2.set(title=f"Średnia {config.VALUE_COL} na rok",
                xlabel="Rok", ylabel=f"Średnia {config.VALUE_COL}")
        return fig1, fig2

    def get_insights(self) -> str:
        """
        Zwraca rok z największą liczbą wydanych gier.

        Zwraca:
            Wnioski jako tekst.
        """
        peak_year = int(self._df["year"].value_counts().idxmax())
        return f"Najwięcej gier wydano w roku {peak_year}."


class ReleaseDayStrategy(AnalysisStrategy):
    """
    Strategia analizy rozkładu premier gier po dniach tygodnia.
    """

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres słupkowy premier gier po dniach tygodnia.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        df = df.dropna(subset=[config.DATE_COL]).copy()
        df["day"] = df[config.DATE_COL].dt.day_name()
        self._df = df

        order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
        counts = df["day"].value_counts().reindex(order, fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set(title="Premiery gier wg dnia tygodnia",
               xlabel="Dzień", ylabel="Liczba gier")
        return fig

    def get_insights(self) -> str:
        """
        Zwraca najpopularniejszy dzień premiery.

        Zwraca:
            Wnioski jako tekst.
        """
        fav = self._df["day"].value_counts().idxmax()
        return f"Najpopularniejszy dzień premiery to {fav}."


class GamesByMonthStrategy(AnalysisStrategy):
    """
    Strategia analizy rozkładu premier gier po miesiącach.
    """

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres słupkowy premier gier po miesiącach.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        df = df.dropna(subset=[config.DATE_COL]).copy()
        df["month"] = df[config.DATE_COL].dt.month_name()
        self._df = df
        order = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
        counts = df["month"].value_counts().reindex(order, fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set(title="Premiery gier wg miesiąca",
               xlabel="Miesiąc", ylabel="Liczba gier")
        return fig

    def get_insights(self) -> str:
        """
        Zwraca miesiąc z największą liczbą premier.

        Zwraca:
            Wnioski jako tekst.
        """
        top_month = (
            self._df[config.DATE_COL]
            .dt.month_name()
            .value_counts()
            .idxmax()
        )
        return f"Najwięcej gier ukazuje się w miesiącu {top_month}."


class AdvancedMultiParamStrategy(AnalysisStrategy):
    """
    Strategia wizualizacji zależności ceny, oceny i liczby recenzji.
    """

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres punktowy cena–ocena, rozmiar bąbla ~ liczba recenzji.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        df = df.dropna(subset=[config.RATING_COL, config.REVIEW_COL,
                               config.VALUE_COL]).copy()
        fig, ax = plt.subplots(figsize=(10, 6))
        sizes = np.sqrt(df[config.REVIEW_COL].clip(lower=1))
        ax.scatter(df[config.VALUE_COL], df[config.RATING_COL],
                   s=sizes, alpha=0.6)
        ax.set(title="Cena vs Ocena (pole bąbla ≈ √recenzji)",
               xlabel="Cena końcowa", ylabel="Pozytywna ocena (%)")
        return fig

    def get_insights(self) -> str:
        """
        Zwraca korelację między ceną a oceną.

        Zwraca:
            Wnioski jako tekst.
        """
        corr, _ = pearsonr(self._df[config.VALUE_COL],
                           self._df[config.RATING_COL])
        direction = "ujemną" if corr < 0 else "dodatnią"
        return f"Korelacja cena–ocena jest {direction} (ρ ≈ {corr:.2f})."


class ThreeDScatterStrategy(AnalysisStrategy):
    """
    Strategia 3D wizualizacji ceny, recenzji i oceny.
    """

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres 3D: cena, log10(recenzji), ocena.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        df = df.dropna(subset=[config.RATING_COL, config.REVIEW_COL,
                               config.VALUE_COL]).copy()
        df = df[df[config.REVIEW_COL] > 0]
        x = df[config.VALUE_COL]
        y = np.log10(df[config.REVIEW_COL])
        z = df[config.RATING_COL]

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(x, y, z, c=z, cmap="viridis",
                        s=20, alpha=0.7, edgecolor="w", linewidth=0.3)
        ax.set(title="3D Cena-Recenzje-Ocena",
               xlabel="Cena", ylabel="log10(Recenzji)",
               zlabel="Ocena (%)")
        ax.view_init(elev=25, azim=140)
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1,
                     label="Ocena (%)")
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95)
        return fig

    def get_insights(self) -> str:
        """
        Zwraca wniosek o przydatności wizualizacji 3D.

        Zwraca:
            Wnioski jako tekst.
        """
        return ("Wykres 3-D jednocześnie wizualizuje cenę, popularność i ocenę – "
                "pozwala wychwycić nietypowe tytuły.")


class CustomScatterStrategy(AnalysisStrategy):
    """
    Strategia do własnych wykresów punktowych między dowolnymi kolumnami.
    """

    def __init__(self, x_col: str, y_col: str, **kw) -> None:
        """
        Inicjalizuje strategię z wybranymi kolumnami x i y.

        Argumenty:
            x_col: Nazwa kolumny osi x.
            y_col: Nazwa kolumny osi y.
        """
        super().__init__(**kw)
        self.x_col = x_col
        self.y_col = y_col

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres punktowy dla wybranych kolumn.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        if self.x_col not in df.columns or self.y_col not in df.columns:
            raise ValueError("Wybrane kolumny nie istnieją w DataFrame.")
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(x=self.x_col, y=self.y_col, data=df,
                        ax=ax, alpha=0.6)
        ax.set(title=f"Własny wykres: {self.x_col} vs {self.y_col}")
        return fig

    def get_insights(self) -> str:
        """
        Zwraca korelację między wybranymi kolumnami.

        Zwraca:
            Wnioski jako tekst.
        """
        d = self._df.dropna(subset=[self.x_col, self.y_col])
        corr, _ = pearsonr(d[self.x_col], d[self.y_col])
        return (f"Korelacja między {self.x_col} a {self.y_col} "
                f"wynosi ρ ≈ {corr:.2f}.")


class TopTagsStrategy(AnalysisStrategy):
    """
    Strategia analizy najczęściej występujących tagów w zbiorze danych.
    """

    def __init__(self, top_n: int = 20, **kw) -> None:
        """
        Inicjalizuje liczbę najpopularniejszych tagów do wyświetlenia.

        Argumenty:
            top_n: Liczba najpopularniejszych tagów.
        """
        super().__init__(**kw)
        self.top_n = top_n

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres słupkowy najczęstszych tagów.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        if "tags" not in df.columns:
            raise ValueError(
                "Brak kolumny 'tags'. Wczytaj dane funkcją load_games_with_metadata()."
            )

        tag_counts = (
            df.explode("tags")
            .loc[lambda d: d["tags"].notna(), "tags"]
            .value_counts()
            .head(self.top_n)
        )

        df_tags = tag_counts.reset_index()
        df_tags.columns = ["name", "count"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelrotation=45)

        sns.barplot(data=df_tags, x="name", y="count", ax=ax, orientation="vertical")
        ax.set(
            title=f"Top {self.top_n} tagów Steam",
            xlabel="Tag", ylabel="Liczba gier"
        )
        return fig

    def get_insights(self) -> str:
        """
        Zwraca najczęściej występujący tag.

        Zwraca:
            Wnioski jako tekst.
        """
        top_tag = (
            self._df.explode("tags")
            .loc[lambda d: d["tags"].notna(), "tags"]
            .value_counts()
            .idxmax()
        )
        return f"Najczęściej występujący tag w wybranym zakresie to **{top_tag}**."


class AvgPriceByTagStrategy(AnalysisStrategy):
    """
    Strategia analizy średniej ceny po tagach.
    """

    def __init__(self, top_n: int = 20, **kw) -> None:
        """
        Inicjalizuje liczbę najpopularniejszych tagów do wyświetlenia.

        Argumenty:
            top_n: Liczba najpopularniejszych tagów.
        """
        super().__init__(**kw)
        self.top_n = top_n

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres słupkowy średniej ceny dla najpopularniejszych tagów.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        if "tags" not in df.columns:
            raise ValueError("Brak kolumny 'tags'. Wczytaj dane funkcją load_games_with_metadata().")

        df_tags = df.dropna(subset=["tags", config.VALUE_COL]).explode("tags")
        avg_price = (
            df_tags.groupby("tags")[config.VALUE_COL]
            .mean()
            .sort_values(ascending=False)
            .head(self.top_n)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=avg_price.values, y=avg_price.index, ax=ax)
        ax.set(title=f"Średnia cena gier dla top {self.top_n} tagów",
               xlabel="Średnia cena", ylabel="Tag")
        return fig

    def get_insights(self) -> str:
        """
        Zwraca tag z najwyższą i najniższą średnią ceną.

        Zwraca:
            Wnioski jako tekst.
        """
        df_tags = (
            self._df.dropna(subset=["tags", config.VALUE_COL])
            .explode("tags")
        )
        grouped = df_tags.groupby("tags")[config.VALUE_COL]
        top_tag = grouped.mean().idxmax()
        top_price = grouped.mean().max()
        min_tag = grouped.mean().idxmin()
        min_price = grouped.mean().min()
        return (
            f"Tag z najwyższą średnią ceną to **{top_tag}** (~{top_price:.2f} zł), "
            f"a najtańszy średnio to **{min_tag}** (~{min_price:.2f} zł)."
        )


class TagCoOccurrenceStrategy(AnalysisStrategy):
    """
    Strategia analizy współwystępowania tagów w grach.
    """

    def __init__(self, top_n: int = 20, **kw) -> None:
        """
        Inicjalizuje liczbę najpopularniejszych par tagów do wyświetlenia.

        Argumenty:
            top_n: Liczba najpopularniejszych par tagów.
        """
        super().__init__(**kw)
        self.top_n = top_n

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres słupkowy współwystępujących par tagów.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        from itertools import combinations
        from collections import Counter

        df_tags = df.dropna(subset=["tags"])
        df_tags = df_tags[df_tags["tags"].apply(lambda x: isinstance(x, list))]

        all_pairs = []
        for tags in df_tags["tags"]:
            tags = list(set(tags))
            all_pairs.extend(combinations(sorted(tags), 2))

        counter = Counter(all_pairs)
        top = counter.most_common(self.top_n)

        if not top:
            raise ValueError("Brak danych do analizy współwystępowania tagów.")

        pairs, counts = zip(*top)
        labels = [f"{a} & {b}" for a, b in pairs]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=counts, y=labels, ax=ax)
        ax.set(title=f"Top {self.top_n} współwystępujących par tagów",
               xlabel="Liczba współwystąpień", ylabel="Para tagów")
        return fig

    def get_insights(self) -> str:
        """
        Zwraca najczęściej współwystępującą parę tagów.

        Zwraca:
            Wnioski jako tekst.
        """
        return (
            f"Najczęściej współwystępujące tagi to **{self._most_common_pair()}**, "
            f"co sugeruje silne powiązanie tematyczne w grach."
        )

    def _most_common_pair(self) -> str:
        """
        Znajdź najczęściej współwystępującą parę tagów w przefiltrowanym DataFrame.

        Zwraca:
            Para tagów jako tekst.
        """
        from itertools import combinations
        from collections import Counter

        df_tags = self._df.dropna(subset=["tags"])
        df_tags = df_tags[df_tags["tags"].apply(lambda x: isinstance(x, list))]

        all_pairs = []
        for tags in df_tags["tags"]:
            tags = list(set(tags))
            all_pairs.extend(combinations(sorted(tags), 2))

        counter = Counter(all_pairs)
        if not counter:
            return "brak danych"
        (a, b), _ = counter.most_common(1)[0]
        return f"{a} & {b}"


class ReviewsOverTimeStrategy(AnalysisStrategy):
    """
    Strategia analizy liczby recenzji w czasie.
    """

    def execute(self, df: pd.DataFrame) -> Figure:
        """
        Rysuje wykres liniowy łącznej liczby recenzji na rok.

        Argumenty:
            df: Wejściowy DataFrame.

        Zwraca:
            Obiekt Figure.
        """
        df = df.dropna(subset=[config.DATE_COL, config.REVIEW_COL]).copy()
        df["year"] = df[config.DATE_COL].dt.year
        self._df = df

        yearly = df.groupby("year")[config.REVIEW_COL].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=yearly, x="year", y=config.REVIEW_COL, marker="o", ax=ax)
        ax.set(title="Łączna liczba recenzji na rok",
               xlabel="Rok", ylabel="Liczba recenzji")
        return fig

    def get_insights(self) -> str:
        """
        Zwraca rok z największą liczbą recenzji i łączną liczbę recenzji.

        Zwraca:
            Wnioski jako tekst.
        """
        yearly_sum = self._df.groupby("year")[config.REVIEW_COL].sum()
        top_year = yearly_sum.idxmax()
        total_reviews = int(yearly_sum.sum())
        return (
            f"Najwięcej recenzji zanotowano w roku {top_year}. "
            f"Łącznie zebrano ponad {total_reviews:,} recenzji."
        )
