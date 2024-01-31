import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the CSV file
dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, 'players/matchups/data/offense/Giannis Antetokounmpo_matchups.csv')


df = pd.read_csv(path)

# Assuming df is your DataFrame with all the data
X = df[['Height Guarded', 'Weight Guarded', 'GuardedHeightDifference', 'GuardedWeightDifference']].values
y = df['Player Points'].values / df['partialPossessions'].values  # This is your PPP

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Define the model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # 4 features to 10
        self.fc2 = nn.Linear(10, 1)  # 10 to 1 output
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = RegressionModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_tensor)
            test_loss = criterion(test_preds.squeeze(), y_test_tensor)
        print(f'Epoch {epoch} | Training Loss: {loss.item()} | Test Loss: {test_loss.item()}')
