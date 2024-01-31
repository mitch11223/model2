import pandas as pd
import os
import json
from Cbase_analyzer import BaseAnalyzer

class PlayerTeamSeasonAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.script_directory)
        print('(7) player_teamanalyzer complete')
        self.gamelogs_path = 'players/gamelogs/'  # Update path as needed
        self.performance_metrics = ['PTS', 'REB', 'AST']
        self.team_metrics = ['W_PCT', 'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 'E_PACE', 'E_REB_PCT']
        self.file_path = 'players/player_json/player_team_weightedaverage.json'
        self.correlation_file_path = 'players/player_json/player_team_correlations.json'
        self.data = self.process_all_players()


    def process_all_players(self):
        results = {}
        correlations = {}

        for filename in os.listdir(self.gamelogs_path):
            if not (filename.startswith('._') or filename.startswith('_.') or filename.startswith('.')):
                file_path = os.path.join(self.gamelogs_path, filename)
                player_name = filename.replace('_log.csv', '').replace('_log.txt', '')
                results[player_name] = self.analyze_player_log(file_path)
                correlations[player_name] = self.calculate_player_correlations(file_path)

        with open(self.file_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
            
        with open(self.correlation_file_path, 'w') as json_file:
            json.dump(correlations, json_file, indent=4)

        return results
    
    def analyze_player_log(self, player_log_path):
        player_log = pd.read_csv(player_log_path)
        analysis_results = {}

        for metric in self.performance_metrics:
            analysis_results[metric] = {}
            for team_metric in self.team_metrics:
                analysis_results[metric][team_metric] = self.calculate_weighted_performance(player_log, metric, team_metric)

        return analysis_results

    def calculate_weighted_performance(self, player_log, player_metric, team_metric):
        weighted_performance = (player_log[player_metric] * player_log[team_metric]).sum()
        normalization_factor = player_log[team_metric].abs().sum()
        normalized_weighted_avg = weighted_performance / normalization_factor if normalization_factor != 0 else 0
        return round(normalized_weighted_avg,3)
    
    def calculate_player_correlations(self, player_log_path):
        player_log = pd.read_csv(player_log_path)
        correlations = {}

        for metric in self.performance_metrics:
            correlations[metric] = {}
            for team_metric in self.team_metrics:
                correlation = self.calculate_correlation(player_log, metric, team_metric)
                correlations[metric][team_metric] = round(correlation, 3)

        return correlations

    def calculate_correlation(self, player_log, player_metric, team_metric):
        correlation = player_log[player_metric].corr(player_log[team_metric])
        return correlation
        
    


            


