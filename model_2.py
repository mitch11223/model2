from Cdata_getter import DataGetter
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()


    def forward(self, x):


