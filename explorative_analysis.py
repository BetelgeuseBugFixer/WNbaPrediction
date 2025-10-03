import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_scatterplot(df, x_column, y_column, title=None, xlabel=None, ylabel=None):
    # Set a visually appealing style for the plots
    sns.set_style("whitegrid")

    # Create the scatter plot
    plt.figure(figsize=(10, 6))  # Set the figure size for better readability
    sns.scatterplot(data=df, x=x_column, y=y_column)

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
    # player_statistics = pd.read_csv("basketballPlayoffs/players.csv")
    # player_statistics["rat"] = player_statistics["o_fgm"]*2 + player_statistics["o_ftm"] + player_statistics["o_3pm"]*3

    team_stats["o_rat"] = ((team_stats["o_fgm"] - team_stats["o_3pm"]) * 2 + team_stats["o_ftm"] + team_stats[
        "o_3pm"] * 3) /(team_stats["o_fga"] + (team_stats["o_fta"]*0.4)+team_stats["o_to"]-team_stats["o_reb"])

    team_stats['d_pos'] = (team_stats['d_fga'] - team_stats['d_oreb'] + team_stats['d_to'] + (0.4 * team_stats['d_fta']))
    team_stats['d_rat'] = (team_stats['d_pts'] / team_stats['d_pos'])
    team_stats['net_ranking'] = team_stats["o_rat"] - team_stats["d_rat"]
    statistics = ["o_rat","d_rat","net_ranking"]

    # for year in range(1,11):
    #    year_data = data[data['years'] == year]
    for statistic in statistics:
        create_scatterplot(team_stats, 'won', statistic)


if __name__ == '__main__':
    main()
