from Cdata_getter import DataGetter
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SingleTaskNet(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(SingleTaskNet, self).__init__()
        self.shared_fc1 = nn.Linear(input_size, 128)
        self.shared_fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x_shared = torch.relu(self.shared_fc1(x))
        x_shared = torch.relu(self.shared_fc2(x_shared))
        x_out = self.fc3(x_shared)
        return x_out

class BaseModel(DataGetter):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        self.get_injuries()
        pd.set_option('display.float_format', '{:.2f}'.format)
        self.scaler = StandardScaler()
        self.net = SingleTaskNet(input_size, output_size)

    def train_model(self, train_loader, criterion, optimizer, num_epochs=1000):
        self.net.train()
        for epoch in range(num_epochs):
            for batch_idx, (data, targets) in enumerate(train_loader):
                predictions = self.net(data)
                loss = criterion(predictions, targets.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test_model(self, test_loader, criterion):
        self.net.eval()
        with torch.no_grad():
            total_loss = 0

            for data, targets in test_loader:
                predictions = self.net(data)
                loss = criterion(predictions, targets.unsqueeze(1))
                total_loss += loss.item()

            avg_loss = total_loss / len(test_loader)
            print(f'Average Loss: {avg_loss:.4f}')


class PlayerModel(BaseModel):
    def __init__(self, input_size=3, output_size=1):
        super().__init__(input_size, output_size)
        self.column_names = ['Opponent Height', 'Opponent Weight', 'Avg Min']
        self.analyze_player_against_team('Damian Lillard','POR')

    def analyze_player_against_team(self, player_name, opp_team):
        self.player_name = player_name
        self.opp_team = opp_team

        player_df = pd.read_csv(f'players/matchups/data/offense/{self.player_name}_matchups.csv')
        player_df = player_df[self.column_names]
        
        
        
        X_scaled = self.scaler.fit_transform(player_df)
        X = X_scaled[:, :-1]
        y = X_scaled[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

        # Define criterion and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        self.train_model(train_loader, criterion, optimizer)
        self.test_model(test_loader, criterion)
        
        roster_df = pd.read_csv('players/matchups/metadata/player_matchups.csv')
        roster_df = roster_df[roster_df['Player Team'] == self.opp_team]
        print(roster_df)
        for team in self.rosters.keys():
            if team in roster_df['Player Team'].unique():
                team_roster = self.rosters[team]
                roster_df = roster_df[roster_df['Player Name'].isin(team_roster.keys())]

        opponent_names = roster_df['Player Name'].values
        roster_df = roster_df[['Player Height', 'Player Weight', 'Avg Min']]
        roster_df.rename(columns={'Player Height': 'Opponent Height', 'Player Weight': 'Opponent Weight'}, inplace=True)

        roster_data_scaled = self.scaler.transform(roster_df[['Opponent Height', 'Opponent Weight', 'Avg Min']])
        roster_data_tensor = torch.tensor(roster_data_scaled, dtype=torch.float32)

        self.net.eval()
        with torch.no_grad():
            predictions = self.net(roster_data_tensor).numpy()

        max_minutes = 0
        max_minutes_player = None
        for name, minutes in zip(opponent_names, predictions):
            minutes = minutes[0]
            if minutes > max_minutes:
                max_minutes = minutes
                max_minutes_player = name

        if max_minutes_player is not None:
            print(f'Player: {player_name}, Highest Projected Matchup Player: {max_minutes_player}, Projected Minutes: {max_minutes}')
        return max_minutes

PlayerModel()