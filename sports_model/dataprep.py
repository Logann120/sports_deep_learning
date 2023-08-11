import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

def main(path = 'nflplaybyplay2009to2016.csv'):
    df = pd.read_csv(path, low_memory=False)
    df_trimmed = df.copy(deep=True)

    # Define the list of columns to keep
    columns_to_keep = [
        'game_id', 'home_team', 'away_team', 'game_date', 'play_type', 'posteam', 'defteam',
        'fourth_down_converted', 'interception', 'own_kickoff_recovery', 'sack', 'punt_blocked',
        'punt_inside_twenty', 'punt_in_endzone', 'yards_gained', 'rush_attempt', 'third_down_converted',
        'drive', 'first_down_rush', 'first_down_pass', 'third_down_failed', 'fourth_down_failed',
        'kickoff_inside_twenty', 'kickoff_in_endzone', 'solo_tackle', 'tackled_for_loss', 'punt_attempt', 'fumble', 'total_home_score', 'total_away_score',
        'complete_pass', 'incomplete_pass'
    ]

    # Select the desired columns from the DataFrame
    df_trimmed = df_trimmed[columns_to_keep]

    # Calculate statistics for each game
    def calculate_game_statistics(game_id, team, is_home):
        filtered_data = df_trimmed[(df_trimmed['game_id'] == game_id) & (df_trimmed['posteam'] == team)]
        if is_home:
            name = 'home_score'
            value = filtered_data['total_home_score'].max()
        else:
            name = 'away_score'
            value = filtered_data['total_away_score'].max()
        statistics = {
            'fourth_down_converted': filtered_data['fourth_down_converted'].sum(),
            'interception': filtered_data['interception'].sum(),
            'own_kickoff_recovery': filtered_data['own_kickoff_recovery'].sum(),
            'sack': filtered_data['sack'].sum(),
            'punt_blocked': filtered_data['punt_blocked'].sum(),
            'punt_inside_twenty': filtered_data['punt_inside_twenty'].sum(),
            'punt_in_endzone': filtered_data['punt_in_endzone'].sum(),
            'yards_gained': filtered_data['yards_gained'].sum(),
            'rush_attempt': filtered_data['rush_attempt'].sum(),
            'third_down_converted': filtered_data['third_down_converted'].sum(),
            'drive': filtered_data['drive'].max() // 2,
            'first_down_rush': filtered_data['first_down_rush'].sum(),
            'first_down_pass': filtered_data['first_down_pass'].sum(),
            'third_down_failed': filtered_data['third_down_failed'].sum(),
            'fourth_down_failed': filtered_data['fourth_down_failed'].sum(),
            'kickoff_inside_twenty': filtered_data['kickoff_inside_twenty'].sum(),
            'kickoff_in_endzone': filtered_data['kickoff_in_endzone'].sum(),
            'solo_tackle': filtered_data['solo_tackle'].sum(),
            'tackled_for_loss': filtered_data['tackled_for_loss'].sum(),
            'punt_attempt': filtered_data['punt_attempt'].sum(),
            'fumble': filtered_data['fumble'].sum(),
            'Pass_comp_percentage': filtered_data['complete_pass'].sum() / (filtered_data['complete_pass'].sum() + filtered_data['incomplete_pass'].sum()),
            name: value
        }
        if not is_home:
            # add a suffix to each key
            statistics = {f'{key}_away': value for key, value in statistics.items()}

        return statistics

    # Process unique game IDs
    unique_game_ids = df_trimmed['game_id'].unique()

    # Initialize lists to store calculated statistics
    game_statistics = []

    # Iterate through unique game IDs and teams
    for game_id in unique_game_ids:
        home_team = df_trimmed[df_trimmed['game_id'] == game_id]['home_team'].iloc[0]
        away_team = df_trimmed[df_trimmed['game_id'] == game_id]['away_team'].iloc[0]
        
        home_statistics = calculate_game_statistics(game_id, home_team, is_home=True)
        away_statistics = calculate_game_statistics(game_id, away_team, is_home=False)
        
        # Append calculated statistics to the list
        game_statistics.append({
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            **home_statistics,
            **away_statistics
        })

    # Create a DataFrame from the calculated statistics
    newdf = pd.DataFrame(game_statistics)
    # Drop unnecessary columns
    to_remove = ['yardline_100', 'quarter_end', 'down', 'goal_to_go', 'ydstogo', 'play_type', 'posteam_score', 
                'defteam_score', 'incomplete_pass', 'penalty', 'pass_attempt', 'complete_pass', 'qtr', 'air_yards', 
                'sp', 'first_down_penalty', 'yards_after_catch']

    #newdf.drop(to_remove, axis=1, inplace=True)
    dates=[]
    ids = newdf.game_id
    for i in ids:
        dates.append(df_trimmed[df_trimmed.game_id == i].reset_index().game_date[0])
    newdf['game_date'] = dates

    df = newdf.copy(deep=True)
    df.home_team = df.home_team.apply(lambda x: 'SD' if x in ['LAC', 'LA'] else 'JAX' if x=='JAC' else x)
    df.away_team = df.away_team.apply(lambda x: 'SD' if x in ['LAC', 'LA'] else 'JAX' if x=='JAC' else x)

    # I noticed we had 35 teams but there are only 32 in the NFL. Some team abbreviations had changed
    # over time, but they're still the same team. The next cell fixes the issue. (SD=LAC=LA) and (JAX=JAC) 
    h_teams = sorted(list(df.home_team.unique()))
    a_teams = sorted(list(df.away_team.unique()))
    team_info=pd.DataFrame(data={'Home Teams': h_teams, 'Away Teams': a_teams})

    # Our response column. Values = 1 if home team wins else 0.
    result: list = []
    for i in df.index:
        if df.home_score[i] > df.away_score_away[i]:
            result.append(1)
        else:
            result.append(0)
    Home_win = pd.Series(data=result, index=df.index, name='Home_win')
    df = pd.concat([df, Home_win], axis=1)
    # Extract the year from the game ID.
    df.game_date = df.game_id.apply(lambda x: int(str(x)[:4]))
    # create list of unique NFL teams
    teams = list(set(h_teams).union(set(a_teams)))
    assert len(teams) == 32, len(teams)
    # There are 32 teams total, so I map them to a numeric value, 0-31. 
    l = [i for i in range(0, 32)]
    dic = dict(zip(teams, l))
    # A list to track each teams wins by year.
    total_wins_per_team_by_year: dict[int, dict[str, int]] = dict((year,dict((key, value) for key, value in zip(
                                            dic.keys(), [0]*len(dic.keys())))) for year in df.game_date.unique())


    # Get the total wins per team by year, and fill in the list.
    for year in df.game_date.unique():
        temp_df = df[df.game_date==year]
        for i in temp_df.index:
            if temp_df.Home_win[i] == 1:
                total_wins_per_team_by_year[year][temp_df.home_team[i]] += 1
            else:
                total_wins_per_team_by_year[year][temp_df.away_team[i]] += 1 


    # Here I am extreacting information to create a feature column. The values will be the ratio
    # of the last seasons total wins for the home team against the away team. Since 2009 is the first
    # year I will just set every value in that year to the average of all other years. 
    result.clear()
    for year in df.game_date.unique():
        df_temp = df[df.game_date==year]
        if not (year == 2009): 
            for i in df_temp.index:
                h_team, a_team = df_temp.home_team[i], df_temp.away_team[i]
                h_wins, a_wins = total_wins_per_team_by_year[year-1][h_team], total_wins_per_team_by_year[year-1][a_team]
                if not (a_wins == 0):
                    result.append({'team': h_team, 'year': year, 'home_win_stat': h_wins/a_wins})
                else: 
                    result.append({'team': h_team, 'year': year, 'home_win_stat': h_wins * 1.5})
        else:
            for i in df_temp.index:
                h_team, a_team = df_temp.home_team[i], df_temp.away_team[i]
                result.append({'team': h_team, 'year': year, 'home_win_stat': 1.0})

    # Create a pd.Series from the list of dictionaries.
    new_feat = pd.DataFrame(result).home_win_stat
    df = pd.concat([df, new_feat], axis=1)
            

    # There were some data entry errors such that the home team was also the away team. I removed
    # these rows.
    df.drop(df.loc[df['home_team']==df['away_team']].index, axis=0, inplace=True)
    # Remove games that ended in a tie.
    idxs = df.loc[df.home_score == df.away_score_away].index
    df.drop(idxs, axis=0, inplace=True)
    df.drop(columns=['game_id', 'home_score', 'away_score_away'], inplace=True)

    # Reorder the columns.
    df = df.reindex(columns=['home_team', 'away_team',
        
        'home_win_stat', 'drive', 'yards_gained','punt_blocked',
        'first_down_rush', 'first_down_pass', 'third_down_converted',
        'third_down_failed', 'fourth_down_converted', 'fourth_down_failed',
        'interception', 'punt_inside_twenty', 'punt_in_endzone',
        'kickoff_inside_twenty', 'kickoff_in_endzone', 'solo_tackle',
        'tackled_for_loss', 'own_kickoff_recovery', 'rush_attempt', 'sack',
        'punt_attempt', 'fumble','Pass_comp_percentage',
            
            
        'drive_away', 'yards_gained_away','punt_blocked_away','first_down_rush_away',
        'first_down_pass_away','third_down_converted_away', 'third_down_failed_away', 'fourth_down_converted_away',
        'fourth_down_failed_away','interception_away','punt_inside_twenty_away', 'punt_in_endzone_away','kickoff_inside_twenty_away',
        'kickoff_in_endzone_away', 'solo_tackle_away', 'tackled_for_loss_away', 'own_kickoff_recovery_away',
        'rush_attempt_away','sack_away','punt_attempt_away','fumble_away','Pass_comp_percentage_away', 
        
        'Home_win'])
    df.reset_index(drop=True, inplace=True)

    # 3-D list with shape 32 (number of teams) X x (nuber of games each team played) X 8 (feature 
                                                                            # columns per team per game)  
    team_stats: list[list] = [[] for i in range(0, 32)]
    # 2-D list to track the win (1) or loss (0) of each team each game.
    team_recent_win_count: list[list] = [[] for i in range(0, 32)]


    # Transform home_team and away_team into numeric columns.
    df.home_team = df.home_team.map(dic)
    df.away_team = df.away_team.map(dic)


    home_cols=[col for col in df.columns[3:-1] if not 'away' in col]
    away_cols=[col for col in df.columns[3:-1] if 'away' in col]
    all_data = []
    for i in range(0, len(df)):
        idx = i
        home_team = df.home_team[idx] # Which team (0-31 is the home team)
        away_team = df.away_team[idx] # Which team (0-31 is the away team)
        
        if i > 47: # Dont add any game stats to the new data frame until each team has played 3 games.
                        # begin adding data on the first instance of a teams fourth game.
            length1 = len(team_stats[home_team])
            # Get recent win count (0-3) for the home team
            last3Home = [team_stats[home_team][length1-3][p] + team_stats[home_team][
                    length1-2][p] + team_stats[home_team][length1-1][p] for p in range(22)]
            length2 = len(team_stats[away_team])
            # Get recent win count (0-3) for the away team
            last3Away = [team_stats[away_team][length2-3][p] + team_stats[away_team][
                    length2-2][p] + team_stats[away_team][length2-1][p] for p in range(22)]
            
            length3 = len(team_recent_win_count[home_team])
            # Record number of recent wins for the current home team.
            last3HWs = sum(team_recent_win_count[home_team][length3-3:])
            length4 = len(team_recent_win_count[away_team])
            # Record number of recent wins for the current away team.
            last3AWs = sum(team_recent_win_count[away_team][length4-3:])
            
            # Add the stats of the last three games (summed) for each team to the dataframe as well
                # as the result of the current game to be used for the predictive model.
            all_data.append([home_team, away_team, last3HWs, df.home_win_stat[i]  # type: ignore
                                    ] + last3Home + last3Away + [last3AWs, df.Home_win[idx]])
        # Store current game stats to be added to the data frame at the teams next "4th" game.
        team_stats[home_team].append(list(df[home_cols].iloc[idx])) # len = 22
        team_stats[away_team].append(list(df[away_cols].iloc[idx])) # len = 22

        # Store the result of the current game to be used in the recent win count at the teams
            # next "4th" game.
        if df.Home_win[idx] == 1:
            team_recent_win_count[home_team].append(1)
            team_recent_win_count[away_team].append(0)
        else:
            team_recent_win_count[home_team].append(0)
            team_recent_win_count[away_team].append(1)    


    # Fill the rows of df2 with the data from all_data.
    df2 = pd.DataFrame(all_data, columns = np.insert(df.columns, [2,-1], ["Recent_wins", 'Recent_wins_away']))
    df2.drop(columns=['home_team', 'away_team'], inplace=True)
    df2.Pass_comp_percentage = df2.Pass_comp_percentage.apply(lambda x: x / 3)
    df2.Pass_comp_percentage_away = df2.Pass_comp_percentage_away.apply(lambda x: x / 3)

    # A function to normalize all the columns.
    def scale_data(dfIn):
        dfIn_norm = dfIn.copy(deep=True)
        for column in dfIn_norm.columns[:-1]:
            dfIn_norm[column] = (dfIn_norm[column] / dfIn_norm[column].max()) * 2 -1 
        return dfIn_norm


    # Create new data frame containing all the normalized values.
    df_normalized = scale_data(df2)
    df_normalized.rename(columns=lambda x: str.capitalize(x), inplace=True)

    # Save new data frame as '.csv' file.
    df_normalized.to_csv('data_final.csv')
    dat = pd.read_csv('data_final.csv')
    dat.drop(columns=['Unnamed: 0'], inplace=True)
    dat_arr = dat.to_numpy(dtype=np.float32, copy=True)
    np.save('data_final.npy', dat_arr)

if __name__ == '__main__':
    path = 'sports_model/nflplaybyplay2009to2016/NFL_Play_by_Play_2009-2018_(v5).csv'
    main(path)