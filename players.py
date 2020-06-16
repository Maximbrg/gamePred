## Fetching data
# Connecting to database
import sqlite3
import pandas as pd
import numpy as np
import datetime

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = "C:\\Users\\shira\\Downloads\\63_589_bundle_archive\\"  # Insert path here
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

def get_fifa_stats(match, player_stats,df):
    ''' Aggregates fifa stats for a given match. '''

    # Define variables
    matchID = match.match_api_id
    date = match['date']
    playersHome = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11"]
    playersAway=["away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    rating = []

    # Loop through all players
    for player in playersHome:

        # Get player ID
        player_id = match[player]

        # Get player stats
        stats = player_stats[player_stats.player_api_id == player_id]

        if np.isnan(player_id) == True:
            overall_rating = 0
            rating.append(overall_rating)

        else:
            overall_rating=stats['overall_rating'].values
            rating.append(overall_rating[0])


    rating.sort(reverse=True)
    ratingForMainRating = rating[:5]
    avgForMainPlayers=np.average(ratingForMainRating)
    avgForAllPlayers=np.average(rating)

    df.loc[df['match_id']==matchID,'avg_performane_of_main_home_players']=avgForMainPlayers
    df.loc[df['match_id']==matchID,'avg_performane_of_all_home_players']=avgForAllPlayers

    print(df[df['match_id']==matchID])
    rating = []
    # Loop through all players
    for player in playersAway:

        # Get player ID
        player_id = match[player]

        # Get player stats
        stats = player_stats[player_stats.player_api_id == player_id]

        if np.isnan(player_id) == True:
            overall_rating = 0
            rating.append(overall_rating)

        else:
            overall_rating=stats['overall_rating'].values
            rating.append(overall_rating[0])


    rating.sort(reverse=True)
    ratingForMainRating = rating[:5]
    avgForMainPlayers=np.average(ratingForMainRating)
    avgForAllPlayers=np.average(rating)

    df.loc[df['match_id']==matchID,'avg_performane_of_main_away_players']=avgForMainPlayers
    df.loc[df['match_id']==matchID,'avg_performane_of_all_away_players']=avgForAllPlayers


table= pd.read_csv("aaa.csv")
df=pd.DataFrame(table)

player_stats = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
player_stats = player_stats.groupby('player_api_id',as_index=False).mean()

match_data = pd.read_sql("SELECT * FROM Match;", conn)
match_data.apply(lambda x :get_fifa_stats(x, player_stats,df),axis=1)
df.to_csv("shiran.csv", index=False)
print(df)
