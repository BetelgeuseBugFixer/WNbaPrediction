import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from baseline_predictor import print_full_table, average_predictor, random_predictor, linear_regression, random_forest, \
    plot_predictions, add_team_rating
from formulas import add_offensive_and_defensive_rating_to_df

TEAM_FILE = "basketballPlayoffs/teams.csv"
PLAYER_FILE = "basketballPlayoffs/players_teams.csv"
WINDOW_SIZE = 3
WIN_PERCENTAGE_BASIC = 0.25
PLAYER_RATING_BASIC = 0.3


def make_standard_team_df_with_win_pct():
    team_df = pd.read_csv(TEAM_FILE)

    # Keep relevant columns
    team_df = team_df[["year", "confID", "tmID", "won", "lost", "GP"]]

    # Add win percentage
    team_df["win_per"] = team_df["won"] / team_df["GP"]

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

    # add conference as bool
    team_df["confID"] = (team_df["confID"] == "EA").astype(int)

    team_df = add_rolling_average_and_previous(team_df, "tmID", WINDOW_SIZE, "win_per", WIN_PERCENTAGE_BASIC)


    # read player file
    player_df = pd.read_csv(PLAYER_FILE)
    player_df = add_offensive_and_defensive_rating_to_df(player_df)
    # add average player ratings
    starting_teams = pd.read_csv(PLAYER_FILE)
    starting_teams = starting_teams[starting_teams["stint"] <= 1]
    player_df = pd.merge(player_df, starting_teams, how="left", on=["playerID", "year"])
    team_rating = add_team_rating(player_df, team_df)

    # add rolling average for team average
    team_rating = add_rolling_average_and_previous(team_rating, "tmID",WINDOW_SIZE,"prev_year_avg_ind" ,PLAYER_RATING_BASIC)
    return team_rating


def main():
    #input_cols_cols = ["win_per_prev", "win_per_rolling", "confID","prev_year_avg_ind_prev","prev_year_avg_ind_rolling"]
    input_cols_cols = ["win_per_prev", "win_per_rolling", "confID"]

    team_df = prepare_df()
    train_df, val_df = split_into_train_and_val(team_df)

    x_train = extract_cols(train_df, input_cols_cols)
    x_val = extract_cols(val_df, input_cols_cols)

    y_train = train_df["win_per"]
    y_val = val_df["win_per"]

    # scale
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    average_pred = average_predictor(y_train, y_val)
    random_pred = random_predictor(y_train, y_val)
    linear_pred = linear_regression(x_train, x_val, y_train)
    rf_pred = random_forest(x_train, x_val, y_train)

    predictions = [average_pred, random_pred, linear_pred, rf_pred]
    algorithms = ["Average", "Random", "Linear Regression", "Random Forest"]
    for prediction, algorithm in zip(predictions, algorithms):
        print(f"{algorithm}:")
        print("MAE:", mean_absolute_error(y_val, prediction))
        print("R²:", r2_score(y_val, prediction))
        plot_predictions(y_val, prediction, algorithm)


if __name__ == '__main__':
    main()
