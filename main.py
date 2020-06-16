## Fetching data
# Connecting to database
import itertools
import sqlite3
import pandas as pd
import datetime

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

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
    match['class'] = results
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


def create_class_column_5lastgames(match_data, df, name):
    scores = [];
    i=0
    for x in df['match_id']:
        match_info = match_data.loc[match_data.match_api_id == x]
        matches_of_team = get_last_matches(match_data, match_info['date'].values[0],
                                           match_info[name].values[0], x=5)
        home_team = match_info[name].values[0]
        score_number = 0
        for y in matches_of_team['match_api_id']:
            match_info_results = df.loc[df.match_id == y]
            result = match_info_results['class'].values[0]
            if match_info_results['away_team_api_id'].values[0] == home_team:
                if result == "Defeat":
                    score_number = score_number + 3
            else:
                if result == "Win":
                    score_number = score_number + 3
            if result == "Draw":
                score_number = score_number + 1
        scores.insert(i, score_number / 15)
        i = i + 1
        print(i)
    df['5Last_Games'+name] = scores


def create_class_5lastgames_between_teams(match_data, df, name):
    scores = [];
    i=0
    for x in df['match_id']:
        match_info = match_data.loc[match_data.match_api_id == x]
        matches_of_team = get_last_matches_against_eachother(match_data, match_info['date'].values[0],
                                                             match_info['home_team_api_id'].values[0],
                                                             match_info['away_team_api_id'].values[0],
                                                             x=5)
        home_team = match_info[name].values[0]
        score_number = 0
        for y in matches_of_team['match_api_id']:
            match_info_results = df.loc[df.match_id == y]
            result = match_info_results['class'].values[0]
            if match_info_results['away_team_api_id'].values[0] == home_team:
                if result == "Defeat":
                    score_number = score_number + 3
            else:
                if result == "Win":
                    score_number = score_number + 3
            if result == "Draw":
                score_number = score_number + 1
        scores.insert(i, score_number / 15)
        i = i + 1
        print(i)
    df['five_last_meetings_for '+name] = scores


path = "C:\\Users\\pc\\Desktop\\"  # Insert path here
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
# d_temp = {'match_id': match_data['match_api_id'].values}
# df = pd.DataFrame(data=d_temp)
# df.insert(1, "home_team_api_id", match_data['home_team_api_id'].values)
# df.insert(2, "away_team_api_id", match_data['away_team_api_id'].values)
# df.insert(3, "home_team_goal", match_data['home_team_goal'].values)
# df.insert(4, "away_team_goal", match_data['away_team_goal'].values)
#
# df = pd.read_csv("aaa.csv")
# create_class_column_5lastgames(match_data, df, 'away_team_api_id')
# create_class_column_5lastgames(match_data, df, 'home_team_api_id')
#
# create_class_5lastgames_between_teams(match_data, df, 'away_team_api_id')
# create_class_5lastgames_between_teams(match_data, df, 'home_team_api_id')

#df.to_csv("final.csv", index=False)

df = pd.read_csv("final.csv")
d_temp = {'5Last_Gamesaway_team_api_id': [], "5Last_Gameshome_team_api_id":[], "last_meetings_for away_team_api_id": [], "five_last_meetings_for home_team_api_id": [],
          'avg_performane_of_main_home_players': [],'avg_performane_of_all_home_players': [],
          'avg_performane_of_main_away_players': [],"avg_performane_of_all_away_players": [], "class": []}
d_temp = {'match_id': match_data['match_api_id'].values}

df_Trn = pd.DataFrame(data=d_temp)
df_Tes = pd.DataFrame(data=d_temp)
print("Start")
match_data2 = match_data[['match_api_id', 'season']]
df = pd.merge(df, match_data2, how='left', on=['match_api_id'])


train = df[~df.season.isin(['2015/2016'])]
test = df[df.season.isin(['2015/2016'])]

X_tr = train[['5Last_Gamesaway_team_api_id','5Last_Gameshome_team_api_id',
               'five_last_meetings_for away_team_api_id',
               'five_last_meetings_for home_team_api_id',
               'avg_performane_of_main_home_players',
               'avg_performane_of_all_home_players',
               'avg_performane_of_main_away_players',
               'avg_performane_of_all_away_players']]
y_tr = train[['class']]

X_test = test[['5Last_Gamesaway_team_api_id','5Last_Gameshome_team_api_id',
               'five_last_meetings_for away_team_api_id',
               'five_last_meetings_for home_team_api_id',
               'avg_performane_of_main_home_players',
               'avg_performane_of_all_home_players',
               'avg_performane_of_main_away_players',
               'avg_performane_of_all_away_players']]
y_test = test[['class']]
# y_pred = SVM.predict(test_x)
#
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test,y_pred))
# print("----------------------------------------------------")
# svclassifier = SVC(kernel='poly', degree=4)
# # print("start")
# # svclassifier.fit(X, y.values.ravel())
# # print("end")
# # y_pred = svclassifier.predict(test_x)
# # print(confusion_matrix(y_test, y_pred))
# # print(classification_report(y_test, y_pred))
print("----------------------RF----------------------------")
RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(X_tr, y_tr.values.ravel())
y_pred = RF.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print("----------------------KNN----------------------------")
KNN_model = KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(X_tr, y_tr.values.ravel())
KNN_prediction = KNN_model.predict(X_test)
print(accuracy_score(KNN_prediction, y_test))
print(classification_report(KNN_prediction, y_test))

print("------------------LR--------------------------------")
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_tr, y_tr.values.ravel())
LR.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

print("---------------------SVM----------------------------")
SVM = svm.SVC(decision_function_shape="ovo").fit(X_tr, y_tr.values.ravel())
SVM.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
# print("---------------------RF----------------------------")
# RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_tr, y_tr.values.ravel())
# RF.predict(X_test)
# print(round(RF.score(X_test, y_test), 4))
# print("--------------------NN------------------------------")
# NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1, max_iter=10000).fit(X_tr, y_tr.values.ravel())
# NN.predict(X_test)
# round(NN.score(X_test, y_test), 4)
# print("----------------------------------------------------")