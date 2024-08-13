import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Base path for the dataset files
BASE_PATH = r"C:\Users\yiyan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Data Analytics And Statistical Learning\Final Project"

def load_dataset(is_training=True):
    """
    Load the dataset from a CSV file.

    Parameters:
    is_training (bool): If True, load the training set; otherwise, load the test set

    Returns:
    pandas.DataFrame: Loaded dataset
    """
    if is_training:
        file_name = "Cleaned_Trojan_Horse_Train_Set_80.csv"
    else:
        file_name = "Cleaned_Trojan_Horse_Test_Set_20.csv"

    file_path = os.path.join(BASE_PATH, file_name)
    return pd.read_csv(file_path)

class TrojanDataset(Dataset):
    """
    TrojanDataset for loading and preprocessing the Trojan Horse dataset.

    Attributes:
    data (pd.DataFrame): The dataset loaded from the CSV file.
    features (pd.DataFrame): Feature columns of the dataset.
    labels (pd.Series): Labels corresponding to the dataset.
    """
    def __init__(self, is_training=True):
        self.data = load_dataset(is_training)
        self.features = self.data.drop('Class', axis=1)
        self.labels = self.data['Class']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Debug: Print column types (comment out in production)
        # if idx == 0:
        #     print("Column types:")
        #     print(self.features.dtypes)

        # Convert features to float, handling non-numeric data
        feature_list = []
        for col in self.features.columns:
            col_data = self.features.iloc[idx][col]
            if pd.api.types.is_numeric_dtype(col_data):
                feature_list.append(float(col_data))
            elif isinstance(col_data, str):
                # Simple hash encoding for string columns
                feature_list.append(float(hash(col_data)) % 1e8)
            else:
                # Placeholder for unhandled data types
                feature_list.append(0.0)

        features = torch.tensor(feature_list, dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return features, label

def get_data_loaders(batch_size=32):
    """
    Create DataLoaders for training and testing datasets.

    Parameters:
    batch_size (int): Batch size for the DataLoaders

    Returns:
    tuple: (train_loader, test_loader)
    """
    train_dataset = TrojanDataset(is_training=True)
    test_dataset = TrojanDataset(is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
