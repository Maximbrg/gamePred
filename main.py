## Fetching data
# Connecting to database
import itertools
import sqlite3
import pandas as pd
import datetime
from sklearn.model_selection import KFold
import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

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
        label.insert(0, 'match_api_id', match['match_api_id'])
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
    i = 0
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
    df['5Last_Games' + name] = scores


def ave_match_in_week(df):
    dfDates = pd.DataFrame(columns=['team_id', 'early_date', 'late_date', 'history_games', 'average_game_per_week'])
    teams = []
    index = 0
    for a in df.itertuples():
        if a.home_team in teams:
            early = dfDates.loc[dfDates['team_id'] == a.home_team]['early_date']
            # print(early.index[0])
            index_early = int(early.index[0])
            date_early_df = datetime.datetime(int(early[index_early][0:4]), int(early[index_early][5:7]),
                                              int(early[index_early][8:10]))
            late = dfDates.loc[dfDates['team_id'] == a.home_team]['late_date']
            index_late = int(late.index[0])
            # print(type(a))
            date_late_df = datetime.datetime(int(late[index_late][0:4]), int(late[index_late][5:7]),
                                             int(late[index_late][8:10]))
            current_date_iter = datetime.datetime(int(a.date[0:4]), int(a.date[5:7]), int(a.date[8:10]))
            if current_date_iter > date_late_df:
                dfDates._set_value(dfDates[dfDates['team_id'] == a.home_team].index.item(), 'late_date',
                                   current_date_iter.strftime("%Y-%m-%d %H:%M:%S"))
                games = dfDates[dfDates['team_id'] == a.home_team]['history_games'].item() + 1
                dfDates._set_value(dfDates[dfDates['team_id'] == a.home_team].index.item(), 'history_games', games)
            elif current_date_iter < date_early_df:
                dfDates._set_value(dfDates[dfDates['team_id'] == a.home_team].index.item(), 'early_date',
                                   current_date_iter.strftime("%Y-%m-%d %H:%M:%S"))
                games = dfDates[dfDates['team_id'] == a.home_team]['history_games'].item() + 1
                dfDates._set_value(dfDates[dfDates['team_id'] == a.home_team].index.item(), 'history_games', games)
            else:
                games = dfDates[dfDates['team_id'] == a.home_team]['history_games'].item() + 1
                dfDates._set_value(dfDates[dfDates['team_id'] == a.home_team].index.item(), 'history_games', games)
        else:
            new_row = {'team_id': a.home_team, 'early_date': a.date, 'late_date': a.date, 'history_games': 1}
            dfDates = dfDates.append(new_row, ignore_index=True)
            teams.append(a.home_team)
        if a.away_team in teams:
            early = dfDates.loc[dfDates['team_id'] == a.away_team]['early_date']
            # print(early.index[0])
            index_early = int(early.index[0])
            date_early_df = datetime.datetime(int(early[index_early][0:4]), int(early[index_early][5:7]),
                                              int(early[index_early][8:10]))
            late = dfDates.loc[dfDates['team_id'] == a.away_team]['late_date']
            index_late = int(late.index[0])
            date_late_df = datetime.datetime(int(late[index_late][0:4]), int(late[index_late][5:7]),
                                             int(late[index_late][8:10]))
            current_date_iter = datetime.datetime(int(a.date[0:4]), int(a.date[5:7]), int(a.date[8:10]))
            if current_date_iter > date_late_df:
                dfDates._set_value(index, 'late_date', current_date_iter.strftime("%Y-%m-%d %H:%M:%S"))
                games = dfDates[dfDates['team_id'] == a.away_team]['history_games'].item() + 1
                dfDates._set_value(dfDates[dfDates['team_id'] == a.away_team].index.item(), 'history_games', games)
            elif current_date_iter < date_early_df:
                dfDates._set_value(index, 'early_date', current_date_iter.strftime("%Y-%m-%d %H:%M:%S"))
                games = dfDates[dfDates['team_id'] == a.away_team]['history_games'].item() + 1
                dfDates._set_value(dfDates[dfDates['team_id'] == a.away_team].index.item(), 'history_games', games)
            else:
                games = dfDates[dfDates['team_id'] == a.away_team]['history_games'].item() + 1
                dfDates._set_value(dfDates[dfDates['team_id'] == a.away_team].index.item(), 'history_games', games)
        else:
            new_row = {'team_id': a.away_team, 'early_date': a.date, 'late_date': a.date, 'history_games': 1}
            dfDates = dfDates.append(new_row, ignore_index=True)
            teams.append(a.away_team)
    for row in dfDates.itertuples():
        early_date = datetime.datetime(int(row.early_date[0:4]), int(row.early_date[5:7]), int(row.early_date[8:10]))
        late_date = datetime.datetime(int(row.late_date[0:4]), int(row.late_date[5:7]), int(row.late_date[8:10]))
        delta = late_date - early_date
        if delta.days == 0:
            ans = 'NaN'
        else:
            number_of_game_history = (dfDates[dfDates['team_id'] == row.team_id]['history_games'].item())
            x = delta.days / number_of_game_history
            ans = 7 / x
        # print(ans)
        dfDates._set_value(dfDates[dfDates['team_id'] == row.team_id].index.item(), 'average_game_per_week', str(ans))
    dfDatesFinal = pd.DataFrame(columns=['team_id', 'average_game_per_week'])
    for row in dfDates.itertuples():
        new_row = {'team_id': row.team_id, 'average_game_per_week': row.average_game_per_week}
        dfDatesFinal = dfDatesFinal.append(new_row, ignore_index=True)

    return dfDatesFinal


def get_last_matches(matches, date, team, x=10):
    ''' Get the last x matches of a given team. '''
    # Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]
    # Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]

    # Return last matches
    return last_matches


def get_average_age_team(df_match, df_players):
    avaragedf = pd.DataFrame(columns=['team_id', 'average_age'])
    teams = []
    teams = []

    players_fields = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
                      "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
                      "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
                      "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
                      "away_player_10", "away_player_11"]
    for row in df_match.itertuples():
        players = []
        if int(row.home_team_api_id) not in teams:
            teams.append(int(row.home_team_api_id))
            matches_of_home_team = get_last_matches(df_match, str(datetime.datetime.now()), row.home_team_api_id, x=10)
            # matches_of_home_team = get_last_matches(df_match, df_match['date'][10001],row.home_team_api_id,x=10)
            for player in players_fields:
                i = 0
                curr_players = matches_of_home_team[player].tolist()
                while i < len(curr_players):
                    if int(curr_players[i]) not in players:
                        players.append(int(curr_players[i]))
                    i = i + 1
            curr_team = row.home_team_api_id
        elif int(row.away_team_api_id) not in teams:
            teams.append(int(row.away_team_api_id))
            matches_of_away_team = get_last_matches(df_match, str(datetime.datetime.now()), row.away_team_api_id, x=10)
            for player in players_fields:
                i = 0
                curr_players = matches_of_home_team[player].tolist()
                while i < len(curr_players):
                    if int(curr_players[i]) not in players:
                        players.append(int(curr_players[i]))
                    i = i + 1
            curr_team = row.away_team_api_id
        sum = 0
        while i < len(players):
            birth_date = df_players[df_players['player_api_id'] == players[i]]['birthday'].item()
            delta = datetime.datetime.now() - datetime.datetime(int(birth_date[0:4]), int(birth_date[5:7]),
                                                                int(birth_date[8:10]))
            sum = sum + (delta.days / 365)
            i = i + 1
        if len(players) > 0:
            average = sum / len(players)
            new_row = {'team_id': int(curr_team), 'average_age': average}
            avaragedf = avaragedf.append(new_row, ignore_index=True)
    return avaragedf


def add_values(df, df_avg_week, df_avg_age):
    for row in df_avg_week.itertuples():
        df.loc[df['home_team_api_id'] == row.team_id, 'home_team_avg_game_week'] = row.average_game_per_week
        df.loc[df['away_team_api_id'] == row.team_id, 'away_team_avg_game_week'] = row.average_game_per_week
    for row in df_avg_age.itertuples():
        df.loc[df['home_team_api_id'] == row.team_id, 'home_team_avg_age'] = row.average_age
        df.loc[df['away_team_api_id'] == row.team_id, 'away_team_avg_age'] = row.average_age
    return (df)


def create_class_5lastgames_between_teams(match_data, df, name):
    scores = [];
    i = 0
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
    df['five_last_meetings_for ' + name] = scores


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
# d_temp = {'match_api_id': match_data['match_api_id'].values}
# df = pd.DataFrame(data=d_temp)
#
# df.insert(1, "home_team_api_id", match_data['home_team_api_id'].values)
# df.insert(2, "away_team_api_id", match_data['away_team_api_id'].values)
# df.insert(3, "home_team_goal", match_data['home_team_goal'].values)
# df.insert(4, "away_team_goal", match_data['away_team_goal'].values)
#
# # Creates target class Win/Defeat/Draw
# get_match_label(df)
#
# # Adding features
# create_class_column_5lastgames(match_data, df, 'away_team_api_id')
# create_class_column_5lastgames(match_data, df, 'home_team_api_id')
#
# create_class_5lastgames_between_teams(match_data, df, 'away_team_api_id')
# create_class_5lastgames_between_teams(match_data, df, 'home_team_api_id')

# df.to_csv("final.csv", index=False)

df = pd.read_csv("final.csv")
d_temp = {'5Last_Gamesaway_team_api_id': [],
          "5Last_Gameshome_team_api_id": [],
          "last_meetings_for_away_team_api_id": [],
          "five_last_meetings_for_home_team_api_id": [],
          'avg_performance_of_main_home_players': [],
          'avg_performance_of_all_home_players': [],
          'avg_performance_of_main_away_players': [],
          "avg_performance_of_all_away_players": [],
          'home_team_avg_game_week': [],
          'away_team_avg_game_week': [],
          'home_team_avg_age': [],
          'away_team_avg_age': [],
          'ave_goal_for_home_team': [],
          'ave_goal_for_away_team': [],
          "class": []}
d_temp = {'match_id': match_data['match_api_id'].values}

df_Trn = pd.DataFrame(data=d_temp)
df_Tes = pd.DataFrame(data=d_temp)
print("Start")
match_data2 = match_data[['match_api_id', 'season']]
df = pd.merge(df, match_data2, how='left', on=['match_api_id'])

df_x = pd.DataFrame(data=d_temp)
df_y = df[['class']]

train = df[~df.season.isin(['2015/2016'])]
test = df[df.season.isin(['2015/2016'])]

X_tr = train[[
    #   '5Last_Gamesaway_team_api_id',
    #  '5Last_Gameshome_team_api_id',
    'five_last_meetings_for_away_team_api_id',
    'five_last_meetings_for_home_team_api_id',
    'avg_performance_of_main_home_players',
    'avg_performance_of_all_home_players',
    'avg_performance_of_main_away_players',
    'avg_performance_of_all_away_players',
    #      'home_team_avg_game_week',
    #       'away_team_avg_game_week',
    # 'home_team_avg_age',
    # 'away_team_avg_age',
    # 'ave_goal_for_home_team',
    # 'ave_goal_for_away_team'
]]
y_tr = train[['class']]
print(X_tr)
X_test = test[[
    #   '5Last_Gamesaway_team_api_id',
    #  '5Last_Gameshome_team_api_id',
    'five_last_meetings_for_away_team_api_id',
    'five_last_meetings_for_home_team_api_id',
    'avg_performance_of_main_home_players',
    'avg_performance_of_all_home_players',
    'avg_performance_of_main_away_players',
    'avg_performance_of_all_away_players',
    #      'home_team_avg_game_week',
    #       'away_team_avg_game_week',
    # 'home_team_avg_age',
    # 'away_team_avg_age',
    # 'ave_goal_for_home_team',
    # 'ave_goal_for_away_team'

]]
y_test = test[['class']]
print()

values = []
print(df.groupby('class')['match_api_id'].nunique())
values.insert(0, 9810 / 21375)
print(9810 / 21375)

print("---------------------RandomForestClassifier----------------------------")
RF = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=3)
RF.fit(X_tr, y_tr.values.ravel())
RF.predict(X_test)
y_pred = RF.predict(X_test)
values.insert(1, accuracy_score(y_pred, y_test))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
# y_test['class2'] = y_pred
# print(y_test)
# y_test.to_csv("finalaa.csv", index=False)
plot_confusion_matrix(RF, X_test, y_test)
plt.title("RandomForestClassifier")

print("----------------------KNeighborsClassifier----------------------------")
KNN_model = KNeighborsClassifier(n_neighbors=330)
KNN_model.fit(X_tr, y_tr.values.ravel())
KNN_prediction = KNN_model.predict(X_test)
values.insert(2, accuracy_score(KNN_prediction, y_test))
print(accuracy_score(KNN_prediction, y_test))
print(classification_report(KNN_prediction, y_test))

plot_confusion_matrix(KNN_model, X_test, y_test)
plt.title("KNeighborsClassifier")

print("---------------------GaussianNB----------------------------")
gnb = GaussianNB()
y_pred = gnb.fit(X_tr, y_tr.values.ravel()).predict(X_test)
values.insert(3, accuracy_score(y_pred, y_test))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(gnb, X_test, y_test)
plt.title("GaussianNB")

print("---------------------LogisticRegression----------------------------")
clf = LogisticRegression(random_state=5).fit(X_tr, y_tr.values.ravel())
y_pred = clf.predict(X_test)
values.insert(4, accuracy_score(y_pred, y_test))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf, X_test, y_test)
plt.title("LogisticRegression")

print("---------------------AdaBoostClassifier----------------------------")
clf = AdaBoostClassifier(n_estimators=100, random_state=5)
y_pred = clf.fit(X_tr, y_tr.values.ravel()).predict(X_test)
values.insert(5, accuracy_score(y_pred, y_test))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf, X_test, y_test)
plt.title("AdaBoostClassifier")

# Graphs

names = ['Apriori', 'RFC', 'KNN', 'NB', 'LR', 'AdaBoostC']
print(values)

plt.subplot(133)
plt.plot(names, values)
# plt.suptitle('Categorical Plotting')

for index, row in y_test.iterrows():
    if row['class'] == "Win":
        row['class'] = 0
    if row['class'] == "Defeat":
        row['class'] = 1
    if row['class'] == "Draw":
        row['class'] = 2

for i, label in enumerate(KNN_prediction):
    if label == "Win":
        KNN_prediction[i] = 0
    if label == "Defeat":
        KNN_prediction[i] = 1
    if label == "Draw":
        KNN_prediction[i] = 2
print(KNN_prediction)
arr = []
arr2 = []
for index, row in df.iterrows():
    arr.append(row['class'])

for i, label in enumerate(arr):
    if label == "Win":
        arr2.append(0)
    if label == "Defeat":
        arr2.append(1)
    if label == "Draw":
        arr2.append(2)
print(arr2)

fig = plt.figure(figsize=(6, 6))
a = fig.add_subplot(xlabel="avg_performance_of_all_home_players", ylabel="avg_performance_of_all_away_players")
a.scatter(X_test['avg_performance_of_all_home_players'], X_test['avg_performance_of_all_away_players'],
          c=y_test['class'])

fig2 = plt.figure(figsize=(6, 6))
b = fig2.add_subplot(xlabel="avg_performance_of_all_home_players", ylabel="avg_performance_of_all_away_players")
b.scatter(X_test['avg_performance_of_all_home_players'], X_test['avg_performance_of_all_away_players'],
          c=KNN_prediction)

fig3 = plt.figure(figsize=(6, 6))
a = fig3.add_subplot(xlabel="five_last_meetings_for_home_team_api_id", ylabel="avg_performance_of_all_home_players")
a.scatter(X_test['five_last_meetings_for_home_team_api_id'], X_test['avg_performance_of_all_home_players'],
          c=y_test['class'])

fig4 = plt.figure(figsize=(6, 6))
b = fig4.add_subplot(xlabel="five_last_meetings_for_home_team_api_id", ylabel="avg_performance_of_all_home_players")
b.scatter(X_test['five_last_meetings_for_home_team_api_id'], X_test['avg_performance_of_all_home_players'],
          c=KNN_prediction)

fig6 = plt.figure(figsize=(6, 6))
b = fig6.add_subplot(xlabel="ave_goal_for_home_team", ylabel="ave_goal_for_away_team")
b.scatter(df['ave_goal_for_home_team'], df['ave_goal_for_away_team'], c=arr2)

fig7 = plt.figure(figsize=(6, 6))
b = fig7.add_subplot(xlabel="home_team_avg_age", ylabel="away_team_avg_age")
b.scatter(df['home_team_avg_age'], df['away_team_avg_age'], c=arr2)


temp3 = pd.crosstab(X_test['five_last_meetings_for_home_team_api_id'], y_test['class'])
temp3.plot(kind='bar', stacked=True, color=['red','blue','black'], grid=False)

temp4 = pd.crosstab(df['ave_goal_for_home_team'], df['class'])
temp4.plot(kind='bar', stacked=True, color=['red','blue','black'], grid=False)

plt.show()
