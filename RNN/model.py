
'''
The purpose of this code is to train a neural network to predict future values 
in a time series dataset, which can be useful in various domains such as 
finance, weather forecasting, or other sequential data problems
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# from stock_data_processing import X_train, X_test, y_train, y_test
from stock_data_processing import X_train, X_test, y_train, y_train_scaled, y_test, y_test_scaled


# convert data to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# create a custom dataset that is used to load data into PyTorch's DataLoader
# by creating this custom dataset we can load data into PyTorch 'DataLoader'
# objects for flexible training and testing
class TimeSeriesDataset(Dataset):
    # Docs explaining what we are doing
    # https://pytorch.org/docs/stable/data.html
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    # initializes input data 'X' and target data 'y'
    def __init__(self, X, y):
        self.X = X
        self.y = y

    # length of data which is the number of samples in dataset
    def __len__(self):
        return len(self.X)

    # called then we index the 'TimeSeriesDataset' with 'idx'
    # represents an index of the dataset. Allows model to access specific
    # examples in the dataset
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.y[idx]

# create train and test Dataloaders
# batch size is the number of samples in each batch of data that will be fed 
# into model during training.
batch_size = 32

# 'train_dataset' and 'test_dataset' are instances of 'TimeSeriesDataset'
# 'train_loader' and 'test_loader' are instances of 'DataLoader' PyTorch class 
# loaders are responsible for iterating over the data in batches.
train_dataset = TimeSeriesDataset(X_train, y_train_scaled)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TimeSeriesDataset(X_test, y_test_scaled)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# defines a feed forward neural network model 
# nn.Module: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
# nn.Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
# nn.ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
# Forward Function: https://pytorch.org/docs/stable/generated/torch.nn.Module.forward.html
class FeedforwardNN(nn.Module):
    # initializes layers of nn model. 
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = 9
hidden_size = 64
output_size = 1

try:
    model = FeedforwardNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load("trained_model.pth"))
    print("Loaded model from trained_model.pth")
except:
    model = FeedforwardNN(input_size, hidden_size, output_size)
    print("Created new model")

# set loss function and optimizer
criterion = nn.MSELoss()
# learning rate is a hyperparameter that determines how much the model's
# parameters are updated during each iteration of training.
optimizer = optim.Adam(model.parameters(), lr=0.002)

# train model
num_epochs = 1000
loss_history = []

for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_history.append(loss.item())
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# evaluate the model
model.eval()
with torch.no_grad():
    total_loss = 0
    num_samples = 0
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item() * batch_X.size(0)
        num_samples += batch_X.size(0)

    avg_loss = total_loss / num_samples
    print(f"Average Test Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "trained_model.pt")

# plot the loss history
plt.plot(loss_history)
plt.title('Model Training Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.show()

