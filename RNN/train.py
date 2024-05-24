import torch
from torch.backends import mps
import torch.nn as nn
import argparse
import numpy as np
import csv

from data_preprocessing import load_data, preprocess_data, split_data
from rnn import RNN
from visualize import plot_loss, plot_combined_predictions, calculate_metrics

def train_model(model, optimizer, criterion, X_train, y_train_scaled, num_epochs, hidden_size, num_layers):
    train_losses = []
    outputs = None
    for epoch in range(num_epochs):
        h = torch.zeros(num_layers, X_train.shape[0], hidden_size)
        optimizer.zero_grad()
        outputs, h = model(X_train, h)
        loss = criterion(outputs, y_train_scaled)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if epoch % 10 == 0:
            print("Epoch: %d, Loss: %1.5f" % (epoch, loss.item()))
    return train_losses, outputs

def test_model(model, criterion, X_test, y_test_scaled, hidden_size):
    with torch.no_grad():
        h = torch.zeros(1, X_test.shape[0], hidden_size)
        test_outputs, h = model(X_test, h)
        test_loss = criterion(test_outputs, y_test_scaled)
        print("Test Loss: %1.5f" % (test_loss.item()))
    return test_loss, test_outputs

def main():
    print("RNN started.")

    # Add argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--hidden_size', type=int, default=64, help="Hidden size of the RNN")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument('--num_layers', type=int, default=1, help="Number of layers in the RNN")
    args = parser.parse_args()

    # Load and preprocess the data
    print("Loading and preprocessing data... ")
    data = load_data('NYSE')
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)
    data, dates, feature_scaler = preprocess_data(data)
    X_train, X_test, y_train, y_test, y_train_scaled, y_test_scaled, dates_train, dates_test, price_scaler = split_data(data, dates)
    print("Data loaded and preprocessed.")

    # Define the model hyperparameters
    print("Defining model hyperparameters... ")
    input_size = X_train.shape[1]
    hidden_size = args.hidden_size
    output_size = 1
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    num_layers = args.num_layers

    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
        device = torch.device('cpu')  # Use CPU if MPS is not available
    else:
        device = torch.device('mps')  # Use MPS if available

    # Initialize the model and optimizer
    model = RNN(input_size, hidden_size, output_size, num_layers=num_layers)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define the loss function
    criterion = nn.MSELoss()

    # Train the model
    print("Training the model...")
    X_train = torch.tensor(X_train.values, dtype=torch.float32, device=device).unsqueeze(1)
    h = torch.zeros(num_layers, X_train.shape[0], hidden_size, device=device)
    y_train_scaled = y_train_scaled.unsqueeze(1).to(device)
    train_losses, outputs = train_model(model, optimizer, criterion, X_train, y_train_scaled, num_epochs, hidden_size, num_layers)
    print("Model training complete. Total training loss: {:.5f}, epochs completed: {}".format(train_losses[-1], len(train_losses)))

    # Test the model
    print("Testing model... ")
    X_test = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_scaled = y_test_scaled.unsqueeze(1).to(device)
    test_loss, test_outputs = test_model(model, criterion, X_test, y_test_scaled, hidden_size)
    test_outputs = test_outputs.detach().cpu()
    print("Model testing complete. Testing loss: {:.5f}".format(test_loss.item()))

    # Save the model
    torch.save(model.state_dict(), 'rnn_model.pth')
    print("Model saved.")

    # Convert the predicted values back to their original scale
    y_train_pred = price_scaler.inverse_transform(outputs.cpu().detach().numpy().reshape(-1, 1))
    y_test_pred = price_scaler.inverse_transform(test_outputs.cpu().detach().numpy().reshape(-1, 1))

    # Convert the dates and prices to NumPy arrays in the appropriate format
    dates_train = dates_train.to_numpy()
    dates_test = dates_test.to_numpy()

    y_train = y_train.to_numpy().reshape(-1)
    y_train_pred = y_train_pred.reshape(-1)
    y_test = y_test.to_numpy().reshape(-1)
    y_test_pred = y_test_pred.reshape(-1)

    # Calculate evaluation metrics
    train_mae, train_mse, train_rmse = calculate_metrics(y_train, y_train_pred)
    test_mae, test_mse, test_rmse = calculate_metrics(y_test, y_test_pred)

    # Save the NumPy arrays to a CSV file
    with open('stock_predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dates Train', 'Y Train', 'Y Train Predicted', 'Dates Test', 'Y Test', 'Y Test Predicted'])
        writer.writerows(zip(dates_train, y_train, y_train_pred, dates_test, y_test, y_test_pred))

    # Mean Absolute Error: metric represents average of the absolute differences
    # between predicted and actual stock prices for the training set. Lower MAE 
    # indicates model's predictions are closer to the actual values in training set
    print("Train MAE:", train_mae)
    # Mean Squared Error: metric represents the average of the squared differences
    # between the predicted and actual stock prices for the training set. More 
    # emphasis on larger errors comparted to MAE. Lower MSE indicates better 
    # performance
    print("Train MSE:", train_mse)
    # Root Mean Squared Error: metric is the square root of the Train MSE. It 
    # represents standard deviation of the prediction errors. How spread out the 
    # errors are from the mean. Lower Train RMSE indicates better performance. 
    # It has the same unit as the target which is easier to interpret
    print("Train RMSE:", train_rmse)
    # Same but it is testing it on the testing data
    print("Test MAE:", test_mae)
    print("Test MSE:", test_mse)
    print("Test RMSE:", test_rmse)
    # compare training and testing metrics to assess if the model is overfitting 
    # or underfitting. Training metrics significantly
        # compare training and testing metrics to assess if the model is overfitting 
    # or underfitting. Training metrics significantly better than testing 
    # metrics it may indicate overfitting(model is too complex and does not 
    # generalize well to unseen data. If training and testing metrics are similar
    # but have high error values, it may indicate underfitting (model is too simple
    # and does not capture underlying patterns

    # Plot the loss and stock price predictions

    np.save("dates_train.npy", dates_train)
    np.save("y_train.npy", y_train)
    np.save("y_train_pred.npy", y_train_pred)
    np.save("dates_test.npy", dates_test)
    np.save("y_test.npy", y_test)
    np.save("y_test_pred.npy", y_test_pred)

    plot_loss(train_losses, [test_loss.item()])
    plot_combined_predictions(dates_train, y_train, y_train_pred, dates_test, y_test, y_test_pred)

if __name__ == '__main__':
   main()

