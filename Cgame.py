from Cdata_getter import DataGetter
import pandas as pd
import numpy as np
import os
import json

class Game(DataGetter):
    def __init__(self,teamA,teamB):
        super().__init__()
        self.teamA = teamA
        self.teamB = teamB
        self.process()
    
    def process(self):
        # Initialize backcourt size for both teams
        backcourt_height_teamA = 0
        backcourt_weight_teamA = 0
        backcourt_height_teamB = 0
        backcourt_weight_teamB = 0

        # Iterate through the players and print individual matchups
        for i, (playerA, playerB) in enumerate(zip(self.teamA, self.teamB)):
            playerA_height = self.players[playerA]['HEIGHT']
            playerA_weight = self.players[playerA]['WEIGHT']
            playerB_height = self.players[playerB]['HEIGHT']
            playerB_weight = self.players[playerB]['WEIGHT']

            # Convert height from cm to inches for the difference
            height_diff_cm = playerA_height - playerB_height
            height_diff_inches = height_diff_cm * 0.393701
            weight_diff = playerA_weight - playerB_weight

            print(f'Matchup: {playerA} vs {playerB}')
            print(f'Height Difference: {height_diff_inches:.2f} inches')
            print(f'Weight Difference: {weight_diff} kg\n')

            # Calculate combined backcourt size for the last two players
            if i >= 3:  # Assuming positions 4 and 5 in the list are backcourt players
                backcourt_height_teamA += playerA_height
                backcourt_weight_teamA += playerA_weight
                backcourt_height_teamB += playerB_height
                backcourt_weight_teamB += playerB_weight

        # Convert the total height difference from cm to inches
        backcourt_height_diff_cm = backcourt_height_teamA - backcourt_height_teamB
        backcourt_height_diff_inches = backcourt_height_diff_cm * 0.393701
        backcourt_weight_diff = backcourt_weight_teamA - backcourt_weight_teamB

        print('Backcourt Size Difference:')
        print(f'Combined Height Difference: {backcourt_height_diff_inches:.2f} inches')
        print(f'Combined Weight Difference: {backcourt_weight_diff} lbs')

    
NOP = ['Jonas Valanciunas','Zion Williamson','Brandon Ingram','Herbert Jones','CJ McCollum']
OKC = ['Chet Holmgren','Jalen Williams','Luguentz Dort','Josh Giddey','Shai Gilgeous-Alexander']

Game(NOP,OKC)