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
    df["o_ind"] = (o_ind_raw / df["minutes"])
    df["d_ind"] = (d_ind_raw / df["minutes"])
    #df["o_ind"] = o_ind_raw
    #df["d_ind"] = d_ind_raw


    df["avg_ind"] = 3*(df["o_ind"] + 2*df["d_ind"]) / 5.0

    # aggreagte by season plus payer
    group_cols = ["playerID", "year"]
    sum_cols = ["GP", "GS", "minutes", "points", "oRebounds", "dRebounds", "rebounds", "assists", "steals", "blocks",
                "turnovers", "PF", "fgAttempted", "fgMade", "ftAttempted", "ftMade", "threeAttempted", "threeMade",
                "dq", "PostGP", "PostGS", "PostMinutes", "PostPoints", "PostoRebounds", "PostdRebounds", "PostRebounds",
                "PostAssists", "PostSteals", "PostBlocks", "PostTurnovers", "PostPF", "PostfgAttempted", "PostfgMade",
                "PostftAttempted", "PostftMade", "PostthreeAttempted", "PostthreeMade", "PostDQ","o_ind","d_ind","avg_ind"]
    agg_dict = {col: "sum" for col in sum_cols}
    df = df.groupby(group_cols, as_index=False).agg(agg_dict)
    return df
