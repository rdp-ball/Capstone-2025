import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Load and preprocess the data
def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Convert 'intention' to numerical values
    intention_mapping = {'cross': 0, 'move along': 1, 'wait': 2}
    df['intention'] = df['intention'].map(intention_mapping)

    # Select features and target
    features = ['position_x', 'position_y', 'speed', 'waiting_time', 'nearest_vehicle_dist', 'nearby_pedestrians']
    X = df[features].values
    y = df['intention'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Define the neural network
class IntentionPredictionNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IntentionPredictionNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Training function
def train_model(model, X_train, y_train, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

class DriverIntentPredictor:
    def __init__(self, model_path=None):
        self.input_size = 6  # position_x, position_y, speed, waiting_time, nearest_vehicle_dist, nearby_pedestrians
        self.hidden_size = 64
        self.num_classes = 3  # cross, move along, wait
        self.model = IntentionPredictionNN(self.input_size, self.hidden_size, self.num_classes)
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def predict(self, features):
        """
        Predict driver intention
        Returns: 0 (cross), 1 (move along), 2 (wait)
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
        # 0: cross
        # 1: move along
        # 2: wait
        return predicted.item()

    def train(self, X_train, y_train, num_epochs=10, learning_rate=0.001, batch_size=32):
        """
        Train the model
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        
        for epoch in range(num_epochs):
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data('pedestrian_intention_dataset_2.csv')

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    # Define model parameters
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 3  # number of intention classes

    # Initialize the model
    model = IntentionPredictionNN(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 1000
    train_model(model, X_train, y_train, criterion, optimizer, num_epochs)

    # Evaluate the model
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Test Accuracy: {accuracy:.2f}')

    # Example usage
    predictor = DriverIntentPredictor()
    X_train, X_test, y_train, y_test = preprocess_data('driver_intention_dataset.csv')
    predictor.train(X_train, y_train)
