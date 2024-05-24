import matplotlib.pyplot as plt
import numpy as np

def load_data_for_visualization():
    dates_train = np.load("dates_train.npy")
    y_train = np.load("y_train.npy")
    y_train_pred = np.load("y_train_pred.npy")
    dates_test = np.load("dates_test.npy")
    y_test = np.load("y_test.npy")
    y_test_pred = np.load("y_test_pred.npy")

    # Load the scaler
    price_scaler = torch.load("price_scaler.pt")

    # Inverse transform the data
    y_train = price_scaler.inverse_transform(y_train.reshape(-1, 1)).squeeze()
    y_train_pred = price_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).squeeze()
    y_test = price_scaler.inverse_transform(y_test.reshape(-1, 1)).squeeze()
    y_test_pred = price_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).squeeze()

    return dates_train, y_train, y_train_pred, dates_test, y_test, y_test_pred


# Plot the training and testing loss
def plot_loss(train_losses, test_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot([0, len(train_losses)], [test_losses[0], test_losses[0]], label='Test Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    plt.show()
    plt.show()

def plot_combined_predictions(dates_train, y_train, y_train_pred, dates_test, y_test, y_test_pred):
    # Print the NumPy arrays
    print("Dates Train:", dates_train)
    print("Y Train:", y_train)
    print("Y Train Predicted:", y_train_pred)
    print("Dates Test:", dates_test)
    print("Y Test:", y_test)
    print("Y Test Predicted:", y_test_pred)

    plt.figure(figsize=(16, 8))

    # Plot actual stock prices
    plt.plot(np.concatenate([dates_train, dates_test]), np.concatenate([y_train, y_test]), label="Actual", linewidth=2)

    # Plot predicted stock prices for training data
    plt.plot(dates_train, y_train_pred, label="Predicted Train", linestyle="--", linewidth=1)

    # Plot predicted stock prices for testing data
    pred_dates = np.concatenate([[dates_train[-1]], dates_test])
    pred_values = np.concatenate([[y_train_pred[-1]], y_test_pred])
    plt.plot(pred_dates, pred_values, label="Predicted Test", linestyle="--", linewidth=1)

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Actual vs Predicted Stock Prices")
    plt.legend()

    plt.show()

# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)

    return mae, mse, rmse

def main():
    dates_train, y_train, y_train_pred, dates_test, y_test, y_test_pred = load_data_for_visualization()
    plot_combined_predictions(dates_train, y_train, y_train_pred, dates_test, y_test, y_test_pred)

if __name__ == "__main__":
    main()

