import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class IntentionPredictionNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(IntentionPredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VehicleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class PedestrianIntentPredictor:
    def __init__(self, model_path=None):
        self.input_size = 6  # step, position_x, position_y, speed, nearest_pedestrian_dist, collision_detected
        self.num_classes = 3  # proceed, slow down, stop
        self.model = IntentionPredictionNN(self.input_size, self.num_classes)
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def preprocess_data(self, data):
        # Ensure data is in the correct format
        if isinstance(data, pd.DataFrame):
            numeric_columns = ['step', 'position_x', 'position_y', 'speed', 'nearest_pedestrian_dist', 'collision_detected']
            X = data[numeric_columns]
        else:
            X = np.array(data).reshape(1, -1)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        return torch.FloatTensor(X_scaled)

    def predict(self, features):
        """
        Predict pedestrian intention
        Returns: 0 (proceed), 1 (slow down), 2 (stop)
        """
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Convert to tensor
        X = torch.FloatTensor(scaled_features)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
            
        # Map neural network output to environment states:
        # 0: proceed
        # 1: slow down
        # 2: stop
        return predicted.item()

    def train(self, train_loader, num_epochs=10, learning_rate=0.001):
        """
        Train the model
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            for i, (features, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
