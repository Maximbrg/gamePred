## Fetching data
# Connecting to database
import sqlite3
import pandas as pd
import datetime

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_match_label(match):
    ''' Derives a label for a given match. '''
    results = []
    for i in range(len(match)):
        print(i)
        home_goals = match['home_team_goal'].values[i]
        away_goals = match['away_team_goal'].values[i]
        label = pd.DataFrame()
        label.insert(0, 'match_id', match['match_id'])
        # Identify match label
        if home_goals > away_goals:
            results.insert(i, "Win")
        if home_goals == away_goals:
            results.insert(i, "Draw")
        if home_goals < away_goals:
            results.insert(i, "Defeat")
    print(results)
    match['labellabel'] = results
    print(match)
    # Return label
    match.to_csv("aaa.csv", index=False)
    return label


def get_last_matches(matches, date, team, x=10):
    ''' Get the last x matches of a given team. '''

    # Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]
    # Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]

    # Return last matches
    return last_matches


def get_last_matches_against_eachother(matches, date, home_team, away_team, x=10):
    ''' Get the last x matches of two given teams. '''

    # Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]
    total_matches = pd.concat([home_matches, away_matches])

    # Get last x matches
    try:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[
                       0:x, :]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[
                       0:total_matches.shape[0], :]

        # Check for error in data
        if (last_matches.shape[0] > x):
            print("Error in obtaining matches")

    # Return data
    return last_matches


path = "C:\\Users\\shira\\Downloads\\63_589_bundle_archive\\"  # Insert path here
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

d = {'home_team_api_id': [], 'away_team_api_id': [], 'home_team_goal': [], 'away_team_goal': [],
     'History_of_5last_games': [],
     'Result_against_for_teams': [], 'Home_game': [], 'ability_front_team': [], 'Average_of_players_age': [],
     'Injuried_main_players': [],
     'Injured_main_players': [], 'ave_match_in_week': [], 'Performance_of_main_players': [],
     'performance_of_all_players': [],
     'ave_goal_in_all_home': [], 'ave_goal_for_Home': []}

# Defining the number of jobs to be run in parallel during grid search
n_jobs = 1  # Insert number of parallel jobs here

# Fetching required data tables
player_data = pd.read_sql("SELECT * FROM Player;", conn)
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
team_data = pd.read_sql("SELECT * FROM Team;", conn)
match_data = pd.read_sql("SELECT * FROM Match;", conn)

# Reduce match data to fulfill run time requirements
rows = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
        "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
        "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
        "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
        "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
        "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]

match_data.dropna(subset=rows, inplace=True)
d_temp = {'match_id': match_data['match_api_id'].values}
df = pd.DataFrame(data=d_temp)
df.insert(1, "home_team_api_id", match_data['home_team_api_id'].values)
df.insert(2, "away_team_api_id", match_data['away_team_api_id'].values)
df.insert(3, "home_team_goal", match_data['home_team_goal'].values)
df.insert(4, "away_team_goal", match_data['away_team_goal'].values)
# print(match_data[:][10001])
print(get_match_label(df))
# History
# print(get_last_matches(match_data, match_data['date'][10001],9987,x=5))

print(get_last_matches_against_eachother(match_data, match_data['date'][10001], match_data['home_team_api_id'][10001],
                                         match_data['away_team_api_id'][10001]))
