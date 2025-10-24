from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from explorative_analysis import create_scatterplot
from formulas import add_offensive_and_defensive_rating_to_df
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def add_team_of_next_season(row, player_team):
    player = row["playerID"]
    year = row["year"]
    return player_team[player][year]


def add_team_rating(player_df, team_df):
    team_starting_squad = defaultdict(lambda: defaultdict(list))
    player_performance_prev_year = defaultdict(dict)
    for row in player_df.itertuples():
        player = row.playerID
        year = int(row.year)
        team_id = row.tmID
        rating = row.avg_ind
        player_performance_prev_year[player][year + 1] = rating
        team_starting_squad[team_id][year].append(player)
    team_average_rating_list = []
    team_id_list = []
    year_list = []
    new_players_list = []
    for team_id in team_starting_squad:
        for year in team_starting_squad[team_id]:
            new_players = 0
            player_ratings = []
            for player in team_starting_squad[team_id][year]:
                rating = player_performance_prev_year.get(player, {}).get(year, None)
                if rating is None:
                    new_players += 1
                    continue
                player_ratings.append(rating)
            team_rating = sum(player_ratings) / len(player_ratings) if player_ratings else None
            team_average_rating_list.append(team_rating)
            year_list.append(year)
            team_id_list.append(team_id)
            new_players_list.append(new_players)

    # create new df for team ratings
    data = {
        "tmID": team_id_list,
        "year": year_list,
        "prev_year_avg_ind": team_average_rating_list,
        "new_players": new_players_list}
    ratings_df = pd.DataFrame(data)
    return pd.merge(ratings_df, team_df, on=["tmID", "year"], how="left")


def calculate_input(player_file, team_file, year):
    player_df = pd.read_csv(player_file)
    team_df = pd.read_csv(team_file)

    # add win percentage
    team_df = team_df[["year", "tmID", "won", "lost", "GP"]]
    team_df["win_per"] = team_df["won"] / team_df["GP"]

    player_df = add_offensive_and_defensive_rating_to_df(player_df)

    starting_teams = pd.read_csv(player_file)
    starting_teams = starting_teams[starting_teams["stint"] <= 1]
    player_df = pd.merge(player_df, starting_teams, how="left", on=["playerID", "year"])

    team_rating = add_team_rating(player_df, team_df)
    # team_rating=player_df.groupby(["tmID","year"])["avg_ind"].mean().reset_index()
    # team_rating.sort_values(by=["tmID","year"], inplace=True)
    # team_rating['Prev_Year_avg_ind']=team_rating.groupby("tmID")["avg_ind"].shift(1)

    # team_df = pd.merge(team_df, team_rating, how="left", on=["tmID", "year"])

    # add previous year performance
    team_rating = team_rating.sort_values(["tmID", "year"])
    team_rating["win_per_prev"] = team_rating.groupby("tmID")["win_per"].shift(1)
    team_rating["no_prev_year"] = team_rating["win_per_prev"].isna().astype(int)
    team_rating["win_per_prev"] = team_rating["win_per_prev"].fillna(0.5)

    # analyse
    create_scatterplot(team_rating, "prev_year_avg_ind", "win_per", hue="win_per_prev")
    print(team_rating[["prev_year_avg_ind", "new_players", "win_per_prev", "win_per"]].corr())
    sns.pairplot(team_rating[["prev_year_avg_ind", "new_players", "win_per_prev", "win_per"]])
    plt.show()
    # split

    train_df = team_rating[team_rating["year"] < 10]
    train_df = train_df[train_df["year"] > 1]

    val_df = team_rating[team_rating["year"] == 10]

    x_train = train_df[["prev_year_avg_ind", "new_players", "win_per_prev"]]
    x_val = val_df[["prev_year_avg_ind", "new_players", "win_per_prev"]]

    # scale
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    # Target
    y_train = train_df["win_per"]
    y_val = val_df["win_per"]
    return x_train, x_val, y_train, y_val


def linear_regression(x_train, x_val, y_train):
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    y_pred = linreg.predict(x_val)
    return y_pred


def random_forest(x_train, x_val, y_train):
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_val)
    return y_pred


def average_predictor(y_train, y_val):
    return [y_train.mean()] * len(y_val)


def random_predictor(y_train, y_val):
    return np.random.uniform(y_train.min(), y_train.max(), size=len(y_val))


def main():
    # get input
    x_train, x_val, y_train, y_val = calculate_input("basketballPlayoffs/players_teams.csv",
                                                     "basketballPlayoffs/teams.csv", 9)
    # apply predictions
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


if __name__ == '__main__':
    main()
