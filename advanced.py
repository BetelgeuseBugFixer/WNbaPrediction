import itertools
import sys
from collections import defaultdict

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from baseline_predictor import print_full_table, average_predictor, random_predictor, linear_regression, random_forest, \
    plot_predictions, add_team_rating, gbr
from formulas import add_offensive_and_defensive_rating_to_df

TEAM_FILE = "basketballPlayoffs/teams.csv"
PLAYER_FILE = "basketballPlayoffs/players_teams.csv"
WINDOW_SIZE = 3
WIN_PERCENTAGE_BASIC = 0.5


def make_standard_team_df_with_win_pct():
    team_df = pd.read_csv(TEAM_FILE)

    # Keep relevant columns
    team_df = team_df[["year", "confID", "tmID", "o_pts", "d_pts", "won", "lost", "GP"]]

    # Add win percentage
    team_df["win_per"] = team_df["won"] / team_df["GP"]

    return team_df


def add_coach_features(team_df, coach_file):
    coaches = pd.read_csv(coach_file)

    coach_yearly = coaches.groupby(["coachID", "year"])[["won", "lost", "post_wins", "post_losses"]].sum().reset_index()
    coach_yearly = coach_yearly.sort_values(["coachID", "year"])

    # History up to NOW
    coach_history = coach_yearly.groupby("coachID")[["won", "lost", "post_wins", "post_losses"]].cumsum()

    # the stats before the season
    coach_history = coach_history.groupby(coach_yearly["coachID"]).shift(1).fillna(0)

    # Add a small epsilon to avoid division by zero for new coaches
    total_games = coach_history["won"] + coach_history["lost"]
    coach_yearly["coach_career_win_per"] = coach_history["won"] / (total_games.replace(0, 1))
    coach_yearly["coach_playoff_wins_prev"] = coach_history["post_wins"]

    # Mark rookies (0 games managed) with a standard win % (e.g. 0.5 or 0.45)
    # If they have 0 games, the division above gave 0. Let's make it 0.5 (neutral assumption)
    is_rookie = total_games == 0
    coach_yearly.loc[is_rookie, "coach_career_win_per"] = 0.5

    # A team might have multiple coaches in one year (stints).
    coaches["games_managed"] = coaches["won"] + coaches["lost"]
    main_coaches = coaches.sort_values("games_managed", ascending=False).drop_duplicates(subset=["tmID", "year"])

    # Merge the career stats we calculated in step 1 into the main_coaches list
    main_coaches = pd.merge(main_coaches,
                            coach_yearly[["coachID", "year", "coach_career_win_per", "coach_playoff_wins_prev"]],
                            on=["coachID", "year"], how="left")

    # Merge into the main team dataframe
    team_df = pd.merge(team_df, main_coaches[["tmID", "year", "coach_career_win_per", "coach_playoff_wins_prev"]],
                       on=["tmID", "year"], how="left")

    # Fill NA for teams where coach data might be missing (default to average)
    team_df["coach_career_win_per"] = team_df["coach_career_win_per"].fillna(0.5)
    team_df["coach_playoff_wins_prev"] = team_df["coach_playoff_wins_prev"].fillna(0)

    return team_df


def add_rolling_average_and_previous(df, id_col, window_size, target_col, standard_val=None):
    prev_col = f"{target_col}_prev"
    roll_col = f"{target_col}_rolling"
    # Sort by team and year for rolling calculations
    df = df.sort_values([id_col, "year"])

    # Previous season win percentage
    df[prev_col] = df.groupby(id_col)[target_col].shift(1)

    # Sort by id and year
    df = df.sort_values([id_col, "year"])

    # Rolling average of previous 3 seasons (per team, no leakage)
    df[roll_col] = df.groupby(id_col)[target_col].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
    )

    # fill na
    if standard_val is not None:
        df[roll_col] = df[roll_col].fillna(standard_val)
        df[prev_col] = df[prev_col].fillna(standard_val)

    return df


def split_into_train_and_val(team_df):
    train_df = team_df[team_df["year"] < 10]
    train_df = train_df[train_df["year"] > 1]

    val_df = team_df[team_df["year"] > 9]

    return train_df, val_df


def extract_cols(team_df, calculated_cols):
    col_df = team_df[calculated_cols]
    return col_df


def prepare_df():
    # prepare df
    team_df = make_standard_team_df_with_win_pct()
    team_df["points_dif"] = team_df["o_pts"] - team_df["d_pts"]
    # add conference as bool
    team_df["confID"] = (team_df["confID"] == "EA").astype(int)

    # calc average first season performance
    min_team_years = team_df.groupby("tmID", as_index=False)["year"].min()
    new_teams_df = pd.merge(team_df, min_team_years, on=["tmID", "year"])

    # Calculate their averages
    new_team_win_per = new_teams_df["win_per"].mean()
    new_team_point_diff = new_teams_df["points_dif"].mean()

    print(f"Baseline Win % (New Teams): {new_team_win_per:.4f}")
    print(f"Baseline Point Diff (New Teams): {new_team_point_diff:.4f}")

    team_df = add_rolling_average_and_previous(team_df, "tmID", WINDOW_SIZE, "win_per", new_team_win_per)

    # add points difference

    team_df = add_rolling_average_and_previous(team_df, "tmID", WINDOW_SIZE, "points_dif", new_team_point_diff)
    # also pythagoras
    team_df["points_dif_pyth"] = team_df["o_pts"] ** 2 / (team_df["o_pts"] ** 2 + team_df["d_pts"] ** 2)
    team_df = add_rolling_average_and_previous(team_df, "tmID", WINDOW_SIZE, "points_dif_pyth", new_team_win_per)

    # read player file
    player_df = pd.read_csv(PLAYER_FILE)
    player_df = add_offensive_and_defensive_rating_to_df(player_df)

    # determine average rookie stat
    min_years = player_df.groupby("playerID", as_index=False)["year"].min()
    new_players_df = pd.merge(player_df, min_years, on=["playerID", "year"])
    new_player_avg = new_players_df["avg_ind"].mean()
    print(f"Calculated New Player Average Rating: {new_player_avg:.4f}")

    # add average player ratings
    starting_teams = pd.read_csv(PLAYER_FILE)
    starting_teams = starting_teams[starting_teams["stint"] <= 1]
    player_df = pd.merge(player_df, starting_teams, how="left", on=["playerID", "year"])
    team_rating = add_team_rating(player_df, team_df)

    # add rolling average for team average
    team_rating = add_rolling_average_and_previous(team_rating, "tmID", WINDOW_SIZE, "prev_year_avg_ind",
                                                   new_player_avg)

    # add coaches
    team_rating = add_coach_features(team_rating, "basketballPlayoffs/coaches.csv")

    # add stable win percentage
    team_rating["stable_win_per"] = team_rating["win_per_prev"] / (1 + 0.5 * team_rating["new_players"])
    team_rating["pyth_adjusted"] = team_rating["points_dif_pyth_rolling"] / (1 + 0.2 * team_rating["new_players"])

    return team_rating


def calculate_score(train_df, val_df, input_cols_cols):
    x_train = extract_cols(train_df, input_cols_cols)
    x_val = extract_cols(val_df, input_cols_cols)

    y_train = train_df["win_per"]
    y_val = val_df["win_per"]

    # scale
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    linear_pred = linear_regression(x_train, x_val, y_train)
    rf_pred = random_forest(x_train, x_val, y_train)
    gbr_pred = gbr(x_train, x_val, y_train)
    predictions = [linear_pred, rf_pred, gbr_pred]
    algorithms = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    res = []
    for prediction, algorithm in zip(predictions, algorithms):
        df_line = [input_cols_cols, algorithm, mean_absolute_error(y_val, prediction), r2_score(y_val, prediction)]
        res.append(df_line)
    return res


def main():
    candidate_cols = [
        "win_per_prev",
        "win_per_rolling",
        "confID",
        "prev_year_avg_ind_prev",
        "prev_year_avg_ind_rolling",
        "points_dif_prev",
        "points_dif_rolling",
        "points_dif_pyth_rolling",
        "points_dif_pyth_prev",
        "new_players",
        "coach_career_win_per",
        "coach_playoff_wins_prev",
        "stable_win_per",
        "pyth_adjusted"
    ]
    team_df = prepare_df()
    train_df, val_df = split_into_train_and_val(team_df)

    results = []
    for r in range(1, 5):
        for combination in itertools.combinations(candidate_cols, r):
            new_df_lines = calculate_score(train_df, val_df, list(combination))
            results += new_df_lines
    # calculate for all cols:
    new_df_lines = calculate_score(train_df, val_df, candidate_cols)
    results += new_df_lines

    result_df_cols = ["features", "Algorithm", "MAE", "R2"]
    df = pd.DataFrame(results, columns=result_df_cols)

    # --- Analysis & Interesting Calcs ---

    print("\n--- Top 5 Configurations by R2 ---")
    # Sort by R2 descending
    print(df.sort_values(by="R2", ascending=False).head(10).to_string(index=False))

    print("\n--- Interesting Calculations ---")

    # 1. Compare Algorithms
    avg_r2 = df.groupby("Algorithm")["R2"].mean()
    print("\nAverage R2 per Algorithm:")
    print(avg_r2)

    # 2. Feature Importance (Frequency in Top 20 models)
    top_20 = df.sort_values(by="R2", ascending=False).head(20)
    feature_counts = defaultdict(int)
    for feats in top_20["features"]:
        for f in feats:
            feature_counts[f] += 1

    print("\nMost Frequent Features in Top 20 Models:")
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    for f, count in sorted_features:
        print(f"{f}: {count}")

    # 3. Best Single Feature
    single_feat_best = df[df["features"].apply(len) == 1].sort_values("R2", ascending=False).head(1)
    print("\nBest Single Feature Model:")
    print(single_feat_best.to_string(index=False))


if __name__ == '__main__':
    main()
