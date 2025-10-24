def add_offensive_and_defensive_rating_to_df(players_teams_stats):
    group_cols = ["playerID", "year"]
    sum_cols=["GP", "GS", "minutes", "points", "oRebounds", "dRebounds", "rebounds", "assists", "steals", "blocks", "turnovers", "PF", "fgAttempted", "fgMade", "ftAttempted", "ftMade", "threeAttempted", "threeMade", "dq", "PostGP", "PostGS", "PostMinutes", "PostPoints", "PostoRebounds", "PostdRebounds", "PostRebounds", "PostAssists", "PostSteals", "PostBlocks", "PostTurnovers", "PostPF", "PostfgAttempted", "PostfgMade", "PostftAttempted", "PostftMade", "PostthreeAttempted", "PostthreeMade", "PostDQ"]
    agg_dict = {col: "sum" for col in sum_cols}
    players_teams_stats = players_teams_stats.groupby(group_cols, as_index=False).agg(agg_dict)
    players_teams_stats["o_ind"] = players_teams_stats["points"] + 1.5 * players_teams_stats["assists"] + 1.2 * players_teams_stats["oRebounds"] \
        + 0.75 * players_teams_stats["threeMade"] - 1.5 * players_teams_stats["turnovers"] \
        - 0.7 * (players_teams_stats["fgAttempted"] - players_teams_stats["fgMade"]) \
        - 0.5 * (players_teams_stats["ftAttempted"] - players_teams_stats["ftMade"])

    players_teams_stats["d_ind"] = players_teams_stats["dRebounds"] + 1.7 * players_teams_stats["steals"] + 1.3 * players_teams_stats["oRebounds"] - 0.8 * players_teams_stats["PF"]

    players_teams_stats["avg_ind"]= (players_teams_stats["o_ind"] + players_teams_stats["d_ind"]) / 2
    return players_teams_stats