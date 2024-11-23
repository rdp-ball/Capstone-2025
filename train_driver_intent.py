import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Main execution
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
