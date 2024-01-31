from Cdata_getter import DataGetter
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#turn offense and defense dfs into numbers/tensors
#choose appropriate neural network for tensors
#leanrs represnetation/patterns/features/weights
#outputs numbers
#turn into human readable fmt




player_name = 'Damian Lillard'
player_team = 'MIL'
opposing_team = 'POR'
offense_cols = ['Opponent Height','Opponent Weight','Avg Min'] #matchup mins here?

offense = pd.read_csv(f'players/matchups/data/offense/{player_name}_matchups.csv')
defense = pd.read_csv(f'players/matchups/data/defense/{player_name}_matchups.csv')


print(offense)