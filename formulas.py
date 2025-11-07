import numpy as np
import pandas as pd

# ...

def add_offensive_and_defensive_rating_to_df(players_teams_stats):
    df = players_teams_stats.copy()

    # ... (o resto do teu pré-cálculo fica igual)

    # --- construir os índices "raw" (como já tens) ---
    o_ind_raw = (
        df["points"]
        + 1.5 * df["assists"]
        + 1.2 * df["oRebounds"]
        + 0.75 * df["threeMade"]
        - 1.5 * df["turnovers"]
        - 0.7 * (df["fgAttempted"] - df["fgMade"])
        - 0.5 * (df["ftAttempted"] - df["ftMade"])
    )

    d_ind_raw = (
       3 * df["dRebounds"]
        + 3 * df["steals"]
        + 1.3 * df["oRebounds"]   # (se quiseres, troca por dRebounds e remove oRebounds aqui)
        - 0.8 * df["PF"]          # atenção: no nosso CSV pode ser 'personalFouls'
    )

    # ✅ 1) garantir que existem no df
    df["o_ind_raw"] = o_ind_raw
    df["d_ind_raw"] = d_ind_raw

    # Helper: z-score por ano a partir de uma Series e do ano
    def z_by_year(series, year_series):
        mu = series.groupby(year_series).transform("mean")
        sd = series.groupby(year_series).transform("std").replace(0, np.nan)
        return (series - mu) / sd

    # ✅ 2) usar a própria Series agrupada por df["year"]
    df["o_ind_z"] = z_by_year(df["o_ind_raw"], df["year"]).fillna(0)
    df["d_ind_z"] = z_by_year(df["d_ind_raw"], df["year"]).fillna(0)

    df["avg_ind"] = 3*(df["o_ind_z"] + 2*df["d_ind_z"]) / 5.0
    return df
