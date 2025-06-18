"""
Moduł GUI – interfejs graficzny do eksploracji i wizualizacji danych o grach Steam.

Zawiera główną klasę okna aplikacji oraz obsługę widżetów, menu i integrację z kontrolerem.
"""
from __future__ import annotations
import logging
import os
from datetime import date
from typing import Dict, List
from tkinter import BOTH, END, LEFT, RIGHT, Y, Toplevel, messagebox as mb, simpledialog

import ttkbootstrap as tb
from ttkbootstrap.constants import PRIMARY, SECONDARY
from ttkbootstrap.widgets import DateEntry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame

import config
import controller
from read_data import load_games_with_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class App(tb.Window):
    """
    Główna klasa aplikacji GUI.

    Odpowiada za inicjalizację okna, ładowanie danych, budowę widżetów
    oraz obsługę interakcji użytkownika.
    """

    df: DataFrame
    min_date: date
    max_date: date

    def __init__(self) -> None:
        """
        Inicjalizuje aplikację i ładuje dane.
        """
        super().__init__(themename="cosmo")
        self.title(config.WINDOW_TITLE)
        self.geometry(config.WINDOW_SIZE)
        self.protocol("WM_DELETE_WINDOW", self._on_exit)

        self.df = load_games_with_metadata()
        self.min_date = self.df[config.DATE_COL].min().date()
        self.max_date = self.df[config.DATE_COL].max().date()

        self._build_widgets()
        self.show_plot()

    def _on_exit(self) -> None:
        """
        Zamyka aplikację.
        """
        try:
            self.quit()
            self.destroy()
        except Exception:
            pass
        os._exit(0)

    def _build_widgets(self) -> None:
        """
        Tworzy wszystkie widżety GUI.
        """
        self._make_menu()

        toolbar = tb.Frame(self)
        toolbar.pack(fill="x", padx=10, pady=6)

        tb.Label(toolbar, text="Analysis:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.analysis_cb = tb.Combobox(
            toolbar,
            values=list(config.ANALYSIS_MAP.keys()),
            bootstyle=SECONDARY,
            state="readonly",
            width=24,
        )
        self.analysis_cb.current(0)
        self.analysis_cb.grid(row=0, column=1, sticky="w", padx=(0, 12))

        plat_frame = tb.Frame(toolbar)
        plat_frame.grid(row=0, column=2, sticky="w", padx=(0, 12))
        self.platform_vars: Dict[str, tb.BooleanVar] = {}
        for p in config.PLATFORM_COLS:
            var = tb.BooleanVar(value=True)
            tb.Checkbutton(plat_frame, text=p.capitalize(), variable=var).pack(side=LEFT, padx=2, pady=1)
            self.platform_vars[p] = var

        price_frame = tb.Frame(toolbar)
        price_frame.grid(row=0, column=3, sticky="w", padx=(0, 12))
        tb.Label(price_frame, text="Price:").pack(side=LEFT)
        self.price_min = tb.Entry(price_frame, width=6)
        self.price_min.pack(side=LEFT, padx=2)
        tb.Label(price_frame, text="-").pack(side=LEFT)
        self.price_max = tb.Entry(price_frame, width=6)
        self.price_max.pack(side=LEFT, padx=2)

        date_frame = tb.Frame(toolbar)
        date_frame.grid(row=0, column=4, sticky="w", padx=(0, 12))
        tb.Label(date_frame, text="Date:").pack(side=LEFT)
        self.start_d = DateEntry(date_frame, width=10, dateformat="%Y-%m-%d", startdate=self.min_date)
        self.start_d.pack(side=LEFT, padx=2, pady=1)
        tb.Label(date_frame, text="-").pack(side=LEFT)
        self.end_d = DateEntry(date_frame, width=10, dateformat="%Y-%m-%d", startdate=self.max_date)
        self.end_d.pack(side=LEFT, padx=2, pady=1)

        tb.Button(toolbar, text="Show", command=self.show_plot, bootstyle=PRIMARY).grid(row=0, column=5, sticky="w")
        toolbar.grid_columnconfigure(6, weight=1)

        self.canvas_frame = tb.Frame(self)
        self.canvas_frame.pack(fill="both", expand=True)

        rpt = tb.Frame(self)
        rpt.pack(fill="x", padx=20, pady=(6, 12))
        self.report = tb.Label(rpt, text="", justify="left", anchor="nw", wraplength=1100)
        self.report.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)

    def _make_menu(self) -> None:
        # Buduje górne menu aplikacji
        menubar = tb.Menu(self)

        theme = tb.Menu(menubar, tearoff=0)
        for th in self.style.theme_names():
            theme.add_command(label=th, command=lambda t=th: self.style.theme_use(t))
        menubar.add_cascade(label="Theme", menu=theme)

        tools = tb.Menu(menubar, tearoff=0)
        tools.add_command(label=config.GET_BY_ID_LABEL, command=self._popup_by_id)
        tools.add_command(label=config.GET_BY_ROW_LABEL, command=self._popup_by_row)
        menubar.add_cascade(label="Tools", menu=tools)

        help_ = tb.Menu(menubar, tearoff=0)
        help_.add_command(label="About", command=lambda: mb.showinfo("About", config.ABOUT_TEXT))
        menubar.add_cascade(label="Help", menu=help_)

        self.config(menu=menubar)

    def _popup_dict(self, data: dict, title: str) -> None:
        # Pokazuje okno z informacją o grze
        top = Toplevel(self)
        top.title(title)
        top.geometry("500x400")
        top.transient(self)

        from tkinter import Scrollbar, Text
        txt = Text(top, wrap="word")
        sb = Scrollbar(top, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        for k, v in data.items():
            txt.insert(END, f"{k:<20}: {v}\n")
        txt.pack(side=LEFT, fill=BOTH, expand=True)
        sb.pack(side=RIGHT, fill=Y)

    def _popup_by_id(self) -> None:
        # Obsługuje wyszukiwanie gry po ID
        val = simpledialog.askstring("Get Game by ID", "Enter ID:", parent=self)
        if val is None:
            return
        d = controller.get_game_info_by_id(self.df, val)
        if d is None:
            mb.showerror("Invalid", config.INVALID_INPUT_MSG)
        elif not d:
            mb.showinfo("Not found", config.NOT_FOUND_MSG)
        else:
            self._popup_dict(d, config.GET_BY_ID_TITLE.format(id=val))

    def _popup_by_row(self) -> None:
        # Obsługuje wyszukiwanie gry po numerze wiersza
        val = simpledialog.askstring("Get Row", "Enter Row #:", parent=self)
        if val is None:
            return
        d = controller.get_game_info_by_row(self.df, val)
        if d is None:
            mb.showerror("Invalid", config.INVALID_INPUT_MSG)
        elif not d:
            mb.showinfo("Not found", config.NOT_FOUND_MSG)
        else:
            self._popup_dict(d, config.GET_BY_ROW_TITLE.format(row=val))

    def show_plot(self) -> None:
        # Generuje i wyświetla wykres na podstawie wybranych parametrów
        key = self.analysis_cb.get()
        plats = [p for p, var in self.platform_vars.items() if var.get()]
        if not plats:
            mb.showerror("Missing", "Select at least one platform")
            return

        start = self.start_d.entry.get().strip() or None
        end = self.end_d.entry.get().strip() or None
        if (msg := controller.validate_dates(self.df, start, end)):
            mb.showerror("Date error", msg)
            return

        pmin = self.price_min.get().strip()
        pmax = self.price_max.get().strip()

        extra: Dict[str, str] = {}
        if key == "Custom 2D scatter":
            x = simpledialog.askstring("Custom scatter", "X column:", parent=self)
            if not x:
                return
            y = simpledialog.askstring("Custom scatter", "Y column:", parent=self)
            if not y:
                return
            extra.update(x_col=x.strip(), y_col=y.strip())

        try:
            figs, insight = controller.run_analysis(
                self.df,
                key,
                plats,
                start,
                end,
                price_min=pmin,
                price_max=pmax,
                **extra,
            )
        except ValueError as ve:
            mb.showerror("Input error", str(ve))
            return
        except Exception as exc:
            logger.exception("run_analysis failed")
            mb.showerror("Execution error", str(exc))
            return

        all_figs: List = []

        def collect(obj):
            # Rekurencyjne zbieranie wszystkich wykresów (jeśli zwrócono kilka)
            if isinstance(obj, (tuple, list)):
                for sub in obj:
                    collect(sub)
            else:
                all_figs.append(obj)

        collect(figs)

        for w in self.canvas_frame.winfo_children():
            w.destroy()
        for fig in all_figs:
            cv = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            cv.draw()
            cv.get_tk_widget().pack(fill="both", expand=True)

        self.report.config(text=insight or "—")


if __name__ == "__main__":
    App().mainloop()
