import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from formulas import add_offensive_and_defensive_rating_to_df


def create_scatterplot(df, x_column, y_column, title=None, xlabel=None, ylabel=None, hue=None):
    # Set a visually appealing style for the plots
    sns.set_style("whitegrid")

    # Create the scatter plot
    plt.figure(figsize=(10, 6))  # Set the figure size for better readability
    sns.scatterplot(data=df, x=x_column, y=y_column,hue=hue)

    plot_title = title if title else f'Scatter Plot of {y_column} vs. {x_column}'
    plt.title(plot_title, fontsize=16)

    x_axis_label = xlabel if xlabel else x_column
    plt.xlabel(x_axis_label, fontsize=12)

    # Use provided y-axis label or default to the column name
    y_axis_label = ylabel if ylabel else y_column
    plt.ylabel(y_axis_label, fontsize=12)

    # Display the plot
    plt.show()


def main():
    team_stats = pd.read_csv("basketballPlayoffs/teams.csv")
    players_stats = pd.read_csv("basketballPlayoffs/players.csv")
    players_teams_stats = pd.read_csv("basketballPlayoffs/players_teams.csv")

    players_teams_stats=add_offensive_and_defensive_rating_to_df(players_teams_stats)
    # player_stats["o_rat"] = player_stats["points"] / (player_stats["o_fgAttempted"] + (player_stats["o_ftAttempted"] * 0.4) + player_stats["o_to"] - player_stats["o_reb"])

    # player_statistics = pd.read_csv("basketballPlayoffs/players.csv")
    # player_statistics["rat"] = player_statistics["o_fgm"]*2 + player_statistics["o_ftm"] + player_statistics["o_3pm"]*3

    team_stats["o_pos"] = (team_stats["o_fga"] + (0.475 * team_stats["o_fta"]) - team_stats["o_reb"] + team_stats["o_to"])
    team_stats["o_rat"] = 100 / team_stats["o_pos"] * team_stats["o_pts"]

    team_stats['d_pos'] = (team_stats['d_fga'] - team_stats['d_oreb'] + (0.475 * team_stats['d_fta'] + team_stats['d_to']))
    team_stats['d_rat'] = (team_stats['d_pts'] / team_stats['d_pos'])

    team_stats['net_rat'] = team_stats["o_rat"] - team_stats["d_rat"]
    statistics = ["o_rat","d_rat","net_rat"]

    players_teams_stats = pd.merge(players_teams_stats, players_stats, left_on='playerID', right_on="bioID", how="inner", validate="m:m")
    players_teams_stats = pd.merge(players_teams_stats, team_stats, on='tmID', how="inner", validate="m:m")

    create_scatterplot(players_teams_stats, 'o_ind', 'd_ind')

    # for year in range(1,11):
    #    year_data = data[data['years'] == year]
    for statistic in statistics:
        create_scatterplot(team_stats, 'won', statistic)

    # plot players performance change from one year to the next


    # check if data is correct


if __name__ == '__main__':
    main()
