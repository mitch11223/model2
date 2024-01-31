from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import leaguestandings
from nba_api.stats.endpoints import teamestimatedmetrics
from nba_api.stats.endpoints import boxscoreadvancedv3
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import boxscoredefensivev2
from nba_api.stats.endpoints import boxscorehustlev2
from nba_api.stats.endpoints import boxscoremiscv3
from nba_api.stats.endpoints import boxscoreplayertrackv3
from nba_api.stats.endpoints import boxscoreusagev3
from nba_api.stats.endpoints import boxscorescoringv3
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import boxscorematchupsv3
from nba_api.live.nba.endpoints import scoreboard

from requests.exceptions import ReadTimeout
from sklearn.preprocessing import StandardScaler
import numpy as np
import glob
import requests
import json
import re
import os
import time
import json
import pandas as pd
import fitz 
from scipy.stats import zscore
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

class DataGetter:
    def __init__(self,execution = 'skip_run'):

            self.date = time.strftime('%m-%d-%Y')
            self.script_directory = os.path.dirname(os.path.abspath(__file__))
            os.chdir(self.script_directory)
            
            #METADATA
            self.players = self.read_players()
            self.rosters = self.read_rosters()
            self.teams = self.read_teams()
            self.game_ids = self.get_gameids()
            self.todays_games = self.get_todays_games()
            self.season = '2023-24'
            self.appened_averages_todict()
            
            if execution == 'skip_run':
                pass
            else:
                self.saveMatchups(mode='offense')
                self.saveMatchups(mode='defense')
                #self.get_injuries()
                
                #GAMELOGS
                self.get_data(player_info_meta = False) #fetches original gamelogs and saves to dir || set player_info_meta to be True for each season
                                  #creates and updates the player_info dict
                               
                #AVERAGES
                self.home_averages_output_folder = 'players/averages/home/'
                self.away_averages_output_folder = 'players/averages/away/'
                self.combined_averages_output_folder = 'players/averages/combined/'
                self.calc_save_averages()
                
                
              
                #DVPOS
                self.api_base_url = "https://api.fantasypros.com/v2/json/nba/team-stats-allowed/"
                self.timeframes = ['7', '15', '30']
                self.acquire_dvpos()
                
                #MEDIANS
                self.acquire_medians()
                
                
                #STANDARD DEVIATION
                self.iterate_through_std()
                
                #TEAM_METRICS
                self.combine_team_dfs()
                
                #PLAYER AND TEAM METRIC MERGER
                self.merge_teamplayer_data()
                
                #NBA API
                self.accessNBA_API_boxscores()
                
                #Update Player Game Logs
                self.add_boxscoregamelog()
                
                #PROPS
                self.props_filename = f"props/{self.date}.txt"
                self.prop_api_key = 'T0RyrHY6WpXU4FYcIGthiwbtBHe0VUHgIdc2VyDO3g'
                self.prop_markets = ["player_points_over_under", "player_rebounds_over_under", "player_assists_over_under"]
                self.prop_data = []
                self.acquire_props(force_update = False)
                
                self.get_teamStats()
                self.add_team_boxscoregamelog()
                
                #matchups
                #self.saveMatchups(mode='offense')
                #self.saveMatchups(mode='defense')

                
    '''
    START
    '''

    
    def get_todays_games(self):
        return scoreboard.ScoreBoard().games.get_dict()
    
    def read_players(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        json_path = os.path.join(dir_path, 'players/player_json/player_info.json')
        with open(json_path) as json_file:
            return json.load(json_file)


    def get_gameids(self):
        game_ids = []
        for filename in os.listdir('players/gamelogs/'):
            file_path = f'players/gamelogs/{filename}'
            if '._' not in file_path:
                try:
                    player_df = pd.read_csv(file_path,delimiter=',')
                    # Use extend to add all Game_ID values to the game_ids list
                    game_ids.extend(player_df['Game_ID'])
                except KeyError:
                    try:
                        player_df = pd.read_csv(file_path,delimiter = '\t')
                        # Use extend to add all Game_ID values to the game_ids list
                        game_ids.extend(player_df['Game_ID'])
                    except KeyError:
                        pass
        
        #game_ids contains every unique Game_ID
        game_ids = list(set(game_ids))


        return game_ids
    
    
    def read_rosters(self):
        # Initialize a dictionary to hold the team rosters
        team_rosters = {}

        # Iterate through each player in the self.players dictionary
        for player_name, player_info in self.players.items():
            # Extract the required information
            player_id = player_info['id']
            team_abbreviation = player_info['TEAM_ABBREVIATION']
            team_id = player_info['TEAM_ID']
            position = player_info['POSITION']

            # Prepare the player's data
            player_data = {
                'id': player_id,
                'TEAMID': team_id,
                'POSITION': position
            }

            # Add the player to the corresponding team in the team_rosters dictionary
            if team_abbreviation not in team_rosters:
                team_rosters[team_abbreviation] = {}
            team_rosters[team_abbreviation][player_name] = player_data
        
        
        with open('teams/metadata/rosters/team_rosters.json','w') as file:
            json.dump(team_rosters,file)
            
        # Convert the team_rosters dictionary to JSON and return it
        return team_rosters
    
    '''
    GAMELOGS (1)
    '''
    
    def build_player_dict(self):
        directory = 'games/2023-24/'
        player_dict = {}

        for filepath in glob.glob(os.path.join(directory, '**/player_advBoxScore.txt'), recursive=True):
            df = pd.read_csv(filepath)

            for _, row in df.iterrows():
                player_id = row['personId']
                full_name = f"{row['firstName']} {row['familyName']}"
                full_name = full_name.replace('.','')
                
                if full_name not in player_dict or player_dict[full_name]['id'] != player_id:
                    player_dict[full_name] = {"id": player_id}

        # Save the dictionary to a file
        with open('players/player_json/player_info.json', 'w') as file:
            json.dump(player_dict, file)
        
        return player_dict

    def player_meta(self, build_player_dict=False):
        if build_player_dict:  # only when new players come into the league.
            self.players = self.build_player_dict()
        else:
            self.players = self.read_players()

        retry_count = 3
        retry_delay = 5
        length = len(self.players)
        count = 0

        for player, attr in self.players.items():
            count += (1/length)
            print(f"{count*100:.2f} %")
            player_id = attr['id']

            for attempt in range(retry_count):
                try:
                    player_info = commonplayerinfo.CommonPlayerInfo(player_id)
                    break
                except ReadTimeout:
                    if attempt < retry_count - 1:
                        print(f"Timeout for {player}. Retrying {attempt + 1}/{retry_count} after {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"Failed to retrieve data for {player} after {retry_count} attempts.")
                        continue  # Skip to the next player after all retries

            meta = player_info.common_player_info.get_data_frame()
            selected_columns = ['HEIGHT', 'WEIGHT', 'BIRTHDATE', 'SEASON_EXP', 'TEAM_ID', 'TEAM_ABBREVIATION', 'POSITION', 'ROSTERSTATUS']
            for column in selected_columns:
                value = str(meta[column].iloc[0])
                try:
                    if column == 'HEIGHT':
                        feet, inches = value.split('-')
                        value = int(feet) * 30.48 + int(inches) * 2.54  # Convert to cm
                        value = round(value, 2)
                    if column == 'WEIGHT':
                        value = float(value)
                    if column == 'BIRTHDATE' and 'T' in str(value):
                        value = value.split('T')[0]
                except ValueError:
                    pass
                attr[column] = value

            # Calculate and add average minutes
            try:
                gamelog_df = pd.read_csv(f'players/averages/combined/{player}_log.txt')
                attr['Avg Min'] = gamelog_df['MIN'].mean()
            except (FileNotFoundError, pd.errors.EmptyDataError):
                print(f"No gamelog data found for {player}.")
                attr['Avg Min'] = None

            time.sleep(1)  # Delay between processing each player

        with open('players/player_json/player_info.json', 'w') as file:
            json.dump(self.players, file)

        print('Player meta completed (0)')
        return self.players

    
    def fetch_player_game_logs(self, player_id, season):
        try:
            game_logs = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            logs_df = game_logs.get_data_frames()[0]
            return logs_df
        except Exception as e:
            return f"Error: {e}"
        
        
    def get_data(self, player_info_meta = False):
        if player_info_meta == True:
            self.players = self.player_meta()
        else:
            self.players = self.read_players()
        
        print('(1) getting gamelogs')
        percentage = 0
        for player, info in self.players.items():
            player_id = info['id']
            time.sleep(0.5)  # Be cautious with using time.sleep in production code
            result = self.fetch_player_game_logs(player_id, '2023-24')
            
            if isinstance(result, pd.DataFrame):
                self.create_cols(result)
                result.to_csv(f"players/gamelogs/{player}_log.txt", sep='\t', index=False)
                percentage += (1/len(self.players))
                print(round(percentage,4),' %')
            else:
                print(f"Error fetching data for {player}: {result}")
        print('Finished player_game_log getter (1)')
        
    def create_cols(self,result):
        result[['Team', 'Location', 'Opponent']] = result['MATCHUP'].str.extract(r'([A-Z]+) ([@vs.]+) ([A-Z]+)')
        return result
    
    '''
    AVERAGES (2)
    '''
    
    def calculate_and_save_averages(self, file_path, game_type):
       
        current_data = pd.read_csv(file_path, delimiter='\t')
        selected_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'PLUS_MINUS']

        if game_type == 'home':
            filtered_data = current_data[current_data['MATCHUP'].str.contains('vs.')]
            output_folder = 'players/averages/home/'
        elif game_type == 'away':
            filtered_data = current_data[current_data['MATCHUP'].str.contains('@')]
            output_folder = 'players/averages/away/'
        else:
            filtered_data = current_data  # For combined, use all data
            output_folder = 'players/averages/combined/'

        column_averages = filtered_data[selected_columns].mean().round(2)
        averages_df = pd.DataFrame([column_averages], columns=selected_columns)
        averages_output_file_path = os.path.join(output_folder, os.path.basename(file_path))
        averages_df.to_csv(averages_output_file_path, index=False)

    def calc_save_averages(self):
        print('average getter')
        for filename in os.listdir('players/gamelogs/'):
            file_path = f'players/gamelogs/{filename}'
            if '._' not in file_path:
                try:
                    self.calculate_and_save_averages(file_path, 'home')
                    self.calculate_and_save_averages(file_path, 'away')
                    self.calculate_and_save_averages(file_path, 'combined')
                except TypeError:
                    pass
                except KeyError:
                    pass
        print('Player home, away, and combined averages calculation complete (2)')
        
    def appened_averages_todict(self):
        for player in self.players:
            try:
                df = pd.read_csv(f'players/averages/combined/{player}_log.txt')
                self.players[player]['AVG_MIN'] = df['MIN'][0]
            except FileNotFoundError:
                pass
        
    '''
    DEFENSE VS POSITION
    '''
    
    def fetch_and_save_dvpos(self, timeframe):
        filename = f"players/defense_vpos/team_defense_vpos_{timeframe}.json"
        params = {'range': timeframe}
        dvpos_headers = {
            'x-api-key': 'CHi8Hy5CEE4khd46XNYL23dCFX96oUdw6qOt1Dnh'  # Be cautious with API keys in code
        }
        response = requests.get(self.api_base_url, headers=dvpos_headers, params=params)

        if response.status_code == 200:
            with open(filename, 'w') as file:
                json.dump(response.json(), file, indent=4)
            #print(f"Data for timeframe {timeframe} saved to {filename}.")
        else:
            print(f"Failed to fetch data for timeframe {timeframe}. Status code: {response.status_code}")

    def acquire_dvpos(self):
        for timeframe in self.timeframes:
            self.fetch_and_save_dvpos(timeframe)
        print("team defense v pos complete (3)")
    
    '''
    MEDIANS (4)
    '''

    def calculate_and_save_medians(self, file_path, game_type):
        current_data = pd.read_csv(file_path, delimiter='\t')
        selected_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'PLUS_MINUS']
        try:
            if game_type == 'home':
                filtered_data = current_data[current_data['MATCHUP'].str.contains('vs.')]
                output_folder = 'players/medians/home/'
            elif game_type == 'away':
                filtered_data = current_data[current_data['MATCHUP'].str.contains('@')]
                output_folder = 'players/medians/away/'
            else:
                filtered_data = current_data  # For combined, use all data
                output_folder = 'players/medians/combined/'

            column_medians = filtered_data[selected_columns].median().round(2)
            medians_df = pd.DataFrame([column_medians], columns=selected_columns)
            medians_output_file_path = os.path.join(output_folder, os.path.basename(file_path))
            medians_df.to_csv(medians_output_file_path, index=False)
        except KeyError:
            pass

    def acquire_medians(self):
        for filename in os.listdir('players/gamelogs/'):
            file_path = f'players/gamelogs/{filename}'
            if '._' not in file_path:
                try:
                    self.calculate_and_save_medians(file_path, 'home')
                    self.calculate_and_save_medians(file_path, 'away')
                    self.calculate_and_save_medians(file_path, 'combined')
                except TypeError:
                    pass
        print('Player home, away, and combined medians calculation complete (4)')

    '''
    PROPS (5)
    '''

    def fetch_game_ids_for_props(self):
        api_date = time.strftime("%Y-%m-%d")
        
        game_endpoint = f"https://api.prop-odds.com/beta/games/nba?date={api_date}&tz=America/New_York&api_key={self.prop_api_key}"
        response = requests.get(game_endpoint)
        if response.status_code == 200:
            data = response.json()
            game_ids = [game["game_id"] for game in data["games"]]
            with open('props/game_ids.txt', 'w') as file:
                json.dump(game_ids, file)  # Provide the file pointer as the first argument
            return game_ids
        else:
            print(f"Failed to fetch game IDs. Status code: {response.status_code}")
            return []


    def fetch_prop_data_for_game(self, game_id):
        for market in self.prop_markets:
            market_endpoint = f'https://api.prop-odds.com/beta/odds/{game_id}/{market}?api_key={self.prop_api_key}'
            resp = requests.get(market_endpoint)
            if resp.status_code == 200:
                market_data = resp.json()['sportsbooks'][2]['market']
                self.prop_data.append(market_data)

    def save_data_to_file(self):
        with open(self.props_filename, 'w') as file:
            json.dump(self.prop_data, file, indent=4)

    def acquire_props(self, force_update=False):
        if not os.path.exists(self.props_filename) or force_update:
            game_ids = self.fetch_game_ids_for_props()
            for game_id in game_ids:
                self.fetch_prop_data_for_game(game_id)
            if self.prop_data:
                self.save_data_to_file()
                print(f"Prop data saved to {self.props_filename}")
            print('Props acquisition complete (5)')
        else:
            print(f"Odds file already exists: {self.props_filename}")
        
    '''
    STANDARD DEVIATION (6)
    '''

    def iterate_through_std(self):
        for filename in os.listdir('players/gamelogs/'):
            file_path = f'players/gamelogs/{filename}'
            if '._' not in file_path:
                self.calculate_and_save_std(file_path)
        print('Standard Deviation Complete (6)')
                    
                    
    def calculate_and_save_std(self, file_path):
        try:
            current_data = pd.read_csv(file_path, delimiter='\t')
            std_devs = current_data[['PTS', 'REB', 'AST']].std()

            # Create a DataFrame correctly
            std_df = pd.DataFrame([std_devs.values], columns=['PTS', 'REB', 'AST'])
            save_path = os.path.join('players/standard_deviations', os.path.basename(file_path))
            std_df.to_csv(save_path, index=False)
            
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    '''
    TEAM METRICS (7)
    '''
    
    def get_team_standings(self):
        standings = leaguestandings.LeagueStandings().get_data_frames()[0]
        standings['TEAM_NAME'] = standings['TeamCity'] + ' ' + standings['TeamName']
        standings = standings.loc[:, ['TEAM_NAME','Conference','Division','WinPCT','HOME','ROAD','PointsPG','OppPointsPG','DiffPointsPG']]
        cols_to_zscore = ['WinPCT','PointsPG','OppPointsPG','DiffPointsPG']
        standings[cols_to_zscore] = zscore(standings[cols_to_zscore])
        
        return standings

    def get_team_metrics(self):
        team_metrics = teamestimatedmetrics.TeamEstimatedMetrics().get_data_frames()[0]
        team_metrics = team_metrics.loc[:, ['TEAM_NAME','W_PCT','E_OFF_RATING','E_DEF_RATING','E_NET_RATING','E_PACE','E_REB_PCT','E_TM_TOV_PCT']]    
        cols_to_zscore = ['W_PCT','E_OFF_RATING','E_DEF_RATING','E_NET_RATING','E_PACE','E_REB_PCT','E_TM_TOV_PCT']
        team_metrics[cols_to_zscore] = zscore(team_metrics[cols_to_zscore])
        team_metrics['E_DEF_RATING'] = -1 * team_metrics['E_DEF_RATING']      
        
        return team_metrics


    def combine_team_dfs(self):
        team_standings = self.get_team_standings()
        team_metrics = self.get_team_metrics()  
        team_data = pd.merge(team_standings, team_metrics, on = 'TEAM_NAME')
        self.abbreviation(team_data)
        team_data.to_csv('teams/metadata/team_data_zscore.csv', index=False)
        print('Team Metrics complete (7)')
    
    def abbreviation(self, df):
        key = {
            'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets', 'CHA': 'Charlotte Hornets',
            'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers', 'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets',
            'DET': 'Detroit Pistons', 'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
            'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat',
            'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks',
            'OKC': 'Oklahoma City Thunder', 'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs', 'TOR': 'Toronto Raptors',
            'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
        }
        df['Team'] = df['TEAM_NAME'].map({v: k for k, v in key.items()})
        return df
    
    '''
    PLAYER AND TEAM METRIC MERGER (8)
    '''
    def merge_teamplayer_data(self):
        team_metrics = pd.read_csv('teams/metadata/team_data_zscore.csv')
        for filename in os.listdir('players/gamelogs/'):
            file_path = os.path.join('players/gamelogs/', filename)
            if '._' or '_.' not in filepath:
                try:
                    player_log = pd.read_csv(file_path,delimiter='\t')
                    for index, row in player_log.iterrows():
                        team_data = team_metrics[team_metrics['Team'] == row['Opponent']] 
                        for col in ['W_PCT','E_OFF_RATING','E_DEF_RATING','E_NET_RATING','E_PACE','E_REB_PCT']:
                            if not team_data.empty:
                                player_log.at[index, col] = team_data.iloc[0][col]
            
                    player_log.to_csv(file_path, index=False)
                    
                except UnicodeDecodeError:
                    pass
        print('Player and Team Metric Merger complete (8)')


    
    
    '''
    NBA API (9)
    '''
    
    def clean_df(self,df):
        columns_to_drop = [col for col in df.columns if '_x' in col or '_y' in col]
        dataframe = df.drop(columns=columns_to_drop)
        dataframe.rename(columns={'playerPoints': 'oppPoints'}, inplace=True)

        
        return dataframe
    
    def accessNBA_API_boxscores(self):
        file_path = 'games/2023-24'
        retry_count = 3
        retry_delay = 5  # Starting delay
        count = 0
        x = 0
        
        for gameid in self.game_ids:
            gameid = f'00{gameid}'
            if os.path.exists(f'{file_path}/{gameid}'):
                print('Game exists: ',gameid, 'Count: ',count)
                count += (1/len(self.game_ids))
                pass
            else:
                for attempt in range(retry_count):
                    try:
                        count += (1/len(self.game_ids))
                        time.sleep(0.5)  # Increasing the delay
                        
                        HUSTLE = boxscorehustlev2.BoxScoreHustleV2(game_id = gameid)
                        DEFENSIVE = boxscoredefensivev2.BoxScoreDefensiveV2(game_id = gameid)
                        ADVANCED = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gameid)
                        MISC = boxscoremiscv3.BoxScoreMiscV3(game_id=gameid)
                        TRACK = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=gameid)
                        USAGE = boxscoreusagev3.BoxScoreUsageV3(game_id=gameid)
                        SCORING = boxscorescoringv3.BoxScoreScoringV3(game_id=gameid)
                
                        player_hustle_stats = HUSTLE.player_stats.get_data_frame()
                        player_defensive_stats = DEFENSIVE.player_stats.get_data_frame()
                        player_advanced_stats = ADVANCED.player_stats.get_data_frame()
                        player_misc_stats = MISC.player_stats.get_data_frame()
                        player_track_stats = TRACK.player_stats.get_data_frame()
                        player_usage_stats = USAGE.player_stats.get_data_frame()
                        player_scoring_stats = SCORING.player_stats.get_data_frame()
                        team_hustle_stats = HUSTLE.team_stats.get_data_frame()
                        team_advanced_stats = ADVANCED.team_stats.get_data_frame()
                        team_misc_stats = MISC.team_stats.get_data_frame()
                        team_track_stats = TRACK.team_stats.get_data_frame()
                        team_usage_stats = USAGE.team_stats.get_data_frame()
                        team_scoring_stats = SCORING.team_stats.get_data_frame()
                        
                        
                        player_df1 = pd.merge(player_hustle_stats,player_defensive_stats, on = 'personId',how = 'outer')
                        player_df2 = pd.merge(player_advanced_stats,player_misc_stats, on = 'personId',how = 'outer')
                        player_df3 = pd.merge(player_track_stats,player_usage_stats, on = 'personId',how = 'outer')
                        player_df = pd.merge(player_df1,player_scoring_stats, on = 'personId',how = 'outer')
                        player_df = pd.merge(player_df,player_df2, on = 'personId',how = 'outer')
                        player_df = pd.merge(player_df,player_df3, on = 'personId',how = 'outer')
                        team_df1 = pd.merge(team_hustle_stats,team_advanced_stats, on = 'teamId',how='outer')
                        team_df2 = pd.merge(team_misc_stats,team_track_stats, on = 'teamId',how='outer')
                        team_df3 = pd.merge(team_usage_stats, team_scoring_stats, on = 'teamId',how='outer')
                        team_df = pd.merge(team_df1, team_df2, on = 'teamId',how='outer')
                        team_df = pd.merge(team_df, team_df3, on = 'teamId',how='outer')
                           
                        player_df = self.clean_df(player_df)
                        team_df = self.clean_df(team_df)

                        os.makedirs(os.path.join(file_path, gameid), exist_ok=True)
                        with open(f'{file_path}/{gameid}/player_BoxScores.txt', 'w') as playerdf_file:
                            player_df.to_csv(playerdf_file, index=False)
                            
                        with open(f'{file_path}/{gameid}/team_BoxScores.txt', 'w') as teamdf_file:
                            team_df.to_csv(teamdf_file, index=False)
                          
                        print('Count: ',count,'Success!')
                        break
                    
                    except ReadTimeout:
                        print(f"Timeout for game ID {gameid}, attempt {attempt + 1}/{retry_count}. Retrying after {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponentially increase the delay
                    except AttributeError:
                        print('Game_ID failed: ',gameid)
                

        print('Game boxscores saved (9)')

    def add_boxscoregamelog(self):
        print('Concatenating player gamelogs and boxscores')
        for filename in os.listdir('players/gamelogs/'):
            file_path = os.path.join('players/gamelogs/', filename)
            if '._' or '_.' not in filepath:
                try:
                    all_rows = []
                    player_log = pd.read_csv(file_path)
                    for index, row in player_log.iterrows():
                        gameid = row['Game_ID']
                        gameid = f'00{gameid}'
                        playerid = row['Player_ID']
                        
                        game_path = f'games/2023-24/{gameid}/player_BoxScores.txt'
                        if os.path.exists(game_path):
                            game = pd.read_csv(game_path)
                            matching_game_row = game[game['personId'] == playerid]

                            if not matching_game_row.empty:
                                merged_row = pd.merge(row.to_frame().T, matching_game_row, left_on='Player_ID', right_on='personId', how='left')
                                all_rows.append(merged_row)
                            else:
                                print(f"No matching row for Player_ID {playerid} in game {gameid}")
                        else:
                            #print(f"Game file not found: {game_path}")
                            pass

                    if all_rows:
                        final_df = pd.concat(all_rows, ignore_index=True)
                        final_df.to_csv(file_path,index=False)
                    else:
                        print(f"No data to concatenate for file {filename}")

                except TypeError as e:
                    print(f"KeyError processing file {filename}: {e}")
                except UnicodeDecodeError:
                    pass
        print('Gamelog and box scores concatenated! (10)')
        
    def read_teams(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        json_path = os.path.join(dir_path, 'teams/metadata/NBA_TeamIds.json')
        with open(json_path) as json_file:
            return json.load(json_file)
        
    def get_teamStats(self):
        directory = 'teams/games/gamelogs'
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        league_gamelog = leaguegamelog.LeagueGameLog(season=self.season).league_game_log.get_data_frame()

        time.sleep(0.5)
        for team, attr in self.teams.items():
            teamid = attr['id']
            team_abbreviation = team
            time.sleep(0.5)
            team_gamelog = teamgamelog.TeamGameLog(season=self.season, team_id=teamid).get_data_frames()[0]
            league_gamelog_filtered = league_gamelog[league_gamelog['TEAM_ABBREVIATION'] != team_abbreviation]
            merged_gamelog = team_gamelog.merge(league_gamelog_filtered, left_on='Game_ID', right_on='GAME_ID', suffixes=('', '_opponent'))

            merged_gamelog['Point_Diff'] = merged_gamelog['PTS'].astype(int) - merged_gamelog['PTS_opponent'].astype(int)
            merged_gamelog['Blowout'] = merged_gamelog['Point_Diff'].apply(lambda x: abs(x) >= 15)
            
            blowout_games = merged_gamelog['Blowout'].sum()
            total_games = len(merged_gamelog)
            blowout_rate = round(((blowout_games / total_games)*100),2)
            

            
            filename = f"{directory}/{team}.csv"
            merged_gamelog.to_csv(filename, index=False)
           


    def add_team_boxscoregamelog(self):
        print('Concatenating team gamelogs and boxscores')
        for filename in os.listdir('teams/games/gamelogs/'):
            file_path = os.path.join('teams/games/gamelogs/', filename)
            if '._' or '_.' not in filepath:
                try:
                    all_rows = []
                    team_log = pd.read_csv(file_path)
                    for index, row in team_log.iterrows():
                        gameid = row['Game_ID']
                        gameid = f'00{gameid}'
                        teamid = row['Team_ID']
                        
                        game_path = f'games/2023-24/{gameid}/team_BoxScores.txt'
                        if os.path.exists(game_path):
                            game = pd.read_csv(game_path)
                            matching_game_row = game[game['teamId'] == teamid]

                            if not matching_game_row.empty:
                                merged_row = pd.merge(row.to_frame().T, matching_game_row, left_on='Team_ID', right_on='teamId', how='left')
                                all_rows.append(merged_row)
                            else:
                                print(f"No matching row for Team_ID {teamid} in game {gameid}")
                        else:
                            #print(f"Game file not found: {game_path}")
                            pass

                    if all_rows:
                        final_df = pd.concat(all_rows, ignore_index=True)
                        final_df.to_csv(file_path,index=False)
                    else:
                        print(f"No data to concatenate for file {filename}")

                except TypeError as e:
                    print(f"KeyError processing file {filename}: {e}")
                except UnicodeDecodeError:
                    pass
        print('Gamelog and box scores concatenated! (11)')
    
    def get_injuries(self):
        site_url = 'https://www.rotowire.com/basketball/tables/injury-report.php'
        params = {'team': 'ALL', 'pos': 'ALL'}
        site_response = requests.get(site_url, params=params)
        inj_json = site_response.json()

        for player in inj_json:
            original_player_name = player['player']
            team, injury, status = player['team'], player['injury'], player['status']

            clean_player_name = original_player_name.replace('.', '')
            if team in self.rosters and clean_player_name in self.rosters[team]:
                if status == 'Out':
                    self.rosters[team].pop(clean_player_name)
            else:
                player_name_with_jr = original_player_name + ' Jr'
                if team in self.rosters and player_name_with_jr in self.rosters[team]:
                    if status == 'Out':
                        self.rosters[team].pop(player_name_with_jr)

    def determine_opp_method(self, partial_possessions):
        if partial_possessions > 25:
            return '1'
        elif 10 <= partial_possessions <= 24:
            return '2'
        else:
            return '3'
        
    def log_game_ids(self, game_ids, log_file='game_ids_log.txt'):
        today_date = datetime.now().strftime('%Y%m%d')
        new_log_file = f'games/metadata/today_gameids.csv'
        with open(new_log_file, 'w') as file:
            file.write(','.join(str(game_ids)) + '\n')

        os.replace(new_log_file, log_file)

    
    def saveMatchups(self, mode='offense'):
        print('selfMatchups Starting')
        all_matchups_df = pd.DataFrame()
        count = 0
        retry_count = 3
        retry_delay = 3
        self.log_game_ids(self.game_ids)

        for game_id in self.game_ids:
            game_id = f'00{game_id}'
            success = False
            count += 1
            print((count/len(self.game_ids)))
            for attempt in range(retry_count):
                try:
                    time.sleep(0.5)
                    game = boxscorematchupsv3.BoxScoreMatchupsV3(game_id=game_id)
                    game_player_stats = game.player_stats.get_data_frame()
                    player_stats_df = self.save_and_print_player_metrics(game_player_stats)
                    all_matchups_df = pd.concat([all_matchups_df, player_stats_df], ignore_index=True)
                    success = True
                    break
                except ReadTimeout:
                    print(f"Timeout for game ID {game_id}, attempt {attempt + 1}/{retry_count}. Retrying after {retry_delay} seconds...")
                    time.sleep(retry_delay)

            if not success:
                print(f"Failed to retrieve data for game ID {game_id} after {retry_count} attempts.")

        try:
            for index, row in all_matchups_df.iterrows():
                player_name = row['Player Name']
                opponent_name = row['Opponent']

                if player_name in self.players:
                    all_matchups_df.at[index, 'Player Height'] = self.players[player_name]['HEIGHT']
                    all_matchups_df.at[index, 'Player Position'] = self.players[player_name]['POSITION']
                    all_matchups_df.at[index, 'Player Weight'] = self.players[player_name]['WEIGHT']

                if opponent_name in self.players:
                    all_matchups_df.at[index, 'Opponent Height'] = self.players[opponent_name]['HEIGHT']
                    all_matchups_df.at[index, 'Opponent Position'] = self.players[opponent_name]['POSITION']
                    all_matchups_df.at[index, 'Opponent Weight'] = self.players[opponent_name]['WEIGHT']
                    all_matchups_df.at[index, 'Avg Min'] = self.players[opponent_name]['Avg Min']

            all_matchups_df['Matchup Minutes'] = all_matchups_df['Matchup Minutes'].apply(lambda x: round(int(x.split(':')[0]) + int(x.split(':')[1]) / 60, 2) if isinstance(x, str) else x)
            all_matchups_df['Player Height'] = pd.to_numeric(all_matchups_df['Player Height'], errors='coerce')
            all_matchups_df['Player Weight'] = pd.to_numeric(all_matchups_df['Player Weight'], errors='coerce')
            all_matchups_df['Opponent Height'] = pd.to_numeric(all_matchups_df['Opponent Height'], errors='coerce')
            all_matchups_df['Opponent Weight'] = pd.to_numeric(all_matchups_df['Opponent Weight'], errors='coerce')
            all_matchups_df['heightDiff'] = round(all_matchups_df['Player Height'] - all_matchups_df['Opponent Height'], 2)
            all_matchups_df['weightDiff'] = all_matchups_df['Player Weight'] - all_matchups_df['Opponent Weight']
            all_matchups_df['ppm'] = round(all_matchups_df['Player Points'] / all_matchups_df['Matchup Minutes'], 2)
            all_matchups_df['ppp'] = round(all_matchups_df['Player Points'] / all_matchups_df['partialPossessions'], 2)
            all_matchups_df['team_ppm'] = round(all_matchups_df['Team Points'] / all_matchups_df['Matchup Minutes'], 2)
            all_matchups_df['team_ppp'] = round(all_matchups_df['Team Points'] / all_matchups_df['partialPossessions'], 2)
            all_matchups_df['POSS_FGA'] = round(all_matchups_df['partialPossessions'] / all_matchups_df['matchupFieldGoalsAttempted'], 2)
        


        except KeyError as e:
            print(f"Key error: {e}")

        player_grouped = all_matchups_df.groupby('Player Name')
        all_matchups_df = all_matchups_df.replace(np.inf, np.nan)
        all_matchups_df.to_csv('players/matchups/data/orig_matchup.csv', index=False)

        player_metrics = {}
        for player in self.players:
            # Filter data for the player
            player_df = all_matchups_df[all_matchups_df['Player Name'] == player]
            opponent_df = all_matchups_df[all_matchups_df['Opponent'] == player]

            total_matchup_mins = player_df['Matchup Minutes'].sum()
            total_possessions = player_df['partialPossessions'].sum()
            total_player_points = player_df['Player Points'].sum()

            total_matchup_mins_opp = opponent_df['Matchup Minutes'].sum()
            total_possessions_opp = opponent_df['partialPossessions'].sum()
            total_player_points_opp = opponent_df['Player Points'].sum()

            offRtg = total_player_points / total_possessions if total_possessions != 0 else 0
            defRtg = total_player_points_opp / total_possessions_opp if total_possessions_opp != 0 else 0

            # average size and weight of people guarded
            height_guarded = sum(
                (opponent_df['partialPossessions'] / total_possessions_opp) * opponent_df['Player Height'])
            weight_guarded = sum(
                (opponent_df['partialPossessions'] / total_possessions_opp) * opponent_df['Player Weight'])

            try:
                player_id = self.players[player]['id']
                player_height = float(self.players[player]['HEIGHT'])
                player_weight = float(self.players[player]['WEIGHT'])
                heightDiff_guarded = player_height - height_guarded
                weightDiff_guarded = player_weight - weight_guarded
                position = self.players[player]['POSITION']

                Avg_Min = self.players[player]['AVG_MIN']
                team = self.players[player]['TEAM_ABBREVIATION']

                player_metrics[player] = {
                    'Player ID': player_id,
                    'Player Team': team,
                    'Position': position,
                    'Avg Min': Avg_Min,
                    'Offensive Rating': round(offRtg, 2),
                    'Defensive Rating': round(defRtg, 2),
                    'Player Height': round(player_height, 2),
                    'Player Weight': round(player_weight, 2),
                    'Height Guarded': round(height_guarded, 2),
                    'Weight Guarded': round(weight_guarded, 2),
                    'GuardedHeightDifference': round(heightDiff_guarded, 2),
                    'GuardedWeightDifference': round(weightDiff_guarded, 2)
                }

                if mode == 'defense':
                    for index, row in opponent_df.iterrows():
                        opponent_name = row['Player Name']
                        opponent_Avg_Min = self.players[opponent_name]['AVG_MIN']
                        # Use opponent's 'AVG_MIN' for defensive dataframe
                        opponent_df.at[index, 'Avg Min'] = opponent_Avg_Min

            except ValueError:
                pass

            except KeyError:
                print(player, 'Error')

            if mode == 'offense':
                player_df.to_csv(f'players/matchups/data/{mode}/{player}_matchups.csv', index=False)
            else:
                opponent_df.to_csv(f'players/matchups/data/{mode}/{player}_matchups.csv', index=False)

        # Convert the dictionary to a DataFrame
        player_metrics_df = pd.DataFrame.from_dict(player_metrics, orient='index')
        player_metrics_df = player_metrics_df[player_metrics_df['Offensive Rating'] != 0]

        # Save the DataFrame to a CSV file
        player_metrics_df.to_csv('players/matchups/metadata/player_matchups.csv', index_label='Player Name')
        
        
        
    def determine_opp_method(self,partial_possessions):
        if partial_possessions > 25:
            return '1'
        elif 10 <= partial_possessions <= 24:
            return '2'
        else:
            return '3'
        
    def save_and_print_player_metrics(self, game_data):
        if not isinstance(game_data, pd.DataFrame):
            raise ValueError("game_data must be a pandas DataFrame")

        columns = ['Game_Id','Player Name', 'Player Position','playerId', 'Opponent', 'Opponent Position','opponentId', 'Matchup Minutes', "partialPossessions", 
                   'Player Points','ppm','ppp', 'Team Points','team_ppm','team_ppp', 'Matchup Assists', 
                   'matchupThreePointersAttempted', 'matchupThreePointersMade', 'matchupFreeThrowsAttempted', 
                   'matchupFieldGoalsMade', 'matchupFieldGoalsAttempted', 'matchupFieldGoalsPercentage', 
          "matchupFreeThrowsMade","shootingFouls",'Player Height','Player Weight','Opponent Height','Opponent Weight', 'heightDiff', 'weightDiff']
        new_rows = []

        for index, record in game_data.iterrows():
            #print(record)
            player_name = f"{record['firstNameOff']} {record['familyNameOff']}"
            player_name = player_name.replace('.','')
            opponent = f"{record['firstNameDef']} {record['familyNameDef']}"
            opponent = opponent.replace('.', '')

            new_row = {
                'Game_Id': record['gameId'],
                'Player Name': player_name,
                'playerId': record['personIdOff'],
                'Opponent': opponent,
                'opponentId': record['personIdDef'],
                'Matchup Minutes': record['matchupMinutes'],
                'partialPossessions': record["partialPossessions"],
                'Player Points': record['playerPoints'],
                'Team Points': record['teamPoints'],
                'Matchup Assists': record['matchupAssists'],
                'matchupThreePointersAttempted': record['matchupThreePointersAttempted'],
                'matchupThreePointersMade': record['matchupThreePointersMade'],
                'matchupFreeThrowsAttempted': record['matchupFreeThrowsAttempted'],
                'matchupFieldGoalsMade': record['matchupFieldGoalsMade'],
                'matchupFieldGoalsAttempted': record['matchupFieldGoalsAttempted'],
                'matchupFieldGoalsPercentage': record['matchupFieldGoalsPercentage'],
                "shootingFouls" : record["shootingFouls"],
                'matchupFreeThrowsMade' : record['matchupFreeThrowsMade']
            }

            new_rows.append(new_row)

        player_stats_df = pd.DataFrame(new_rows, columns=columns)
        return player_stats_df
    

                     
#DataGetter(execution='skip_run')