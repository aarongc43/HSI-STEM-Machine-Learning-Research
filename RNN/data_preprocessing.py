import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    if os.path.isfile(path):
        data = pd.read_csv(path)
        return data
    elif os.path.isdir(path):
        data_frames = []
        for filename in os.listdir(path):
            if filename.endswith('.csv'):
                data = pd.read_csv(os.path.join(path, filename))
                data_frames.append(data)
        return pd.concat(data_frames, axis=0, ignore_index=True)
    else:
        raise ValueError("Provided path is not a file or a directory.")

def preprocess_data(data):
    dates = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data = data.drop('Date', axis=1)

    feature_scaler = MinMaxScaler()
    columns_to_scale = ['Open', 'High', 'Low', 'Year', 'Month', 'Day']
    if 'RSI' in data.columns:
        columns_to_scale.append('RSI')
    if 'Momentum' in data.columns:
        columns_to_scale.append('Momentum')
    if 'Volume' in data.columns:
        columns_to_scale.append('Volume')
    data[columns_to_scale] = feature_scaler.fit_transform(data[columns_to_scale])

    return data, dates, feature_scaler

def split_data(data, dates, test_size=0.2, random_state=42):
    np.random.seed(random_state)

    X = data.drop('Close', axis=1)
    y = data['Close']

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=test_size, random_state=random_state, shuffle=False)

    feature_scaler = MinMaxScaler()
    X_train = pd.DataFrame(feature_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(feature_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    price_scaler = MinMaxScaler()
    y_train_scaled = torch.tensor(price_scaler.fit_transform(y_train.values.reshape(-1, 1)), dtype=torch.float32).squeeze()
    y_test_scaled = torch.tensor(price_scaler.transform(y_test.values.reshape(-1, 1)), dtype=torch.float32).squeeze()

    y_train = pd.DataFrame(y_train_scaled.numpy(), columns=['Close'], index=y_train.index)
    y_test = pd.DataFrame(y_test_scaled.numpy(), columns=['Close'], index=y_test.index)

    dates_train = dates.iloc[X_train.index]
    dates_test = dates.iloc[X_test.index]
    
    return X_train, X_test, y_train, y_test, y_train_scaled, y_test_scaled, dates_train, dates_test, price_scaler

data_path = 'NYSE'
data = load_data(data_path)
data.reset_index(drop=True, inplace=True)
data, dates, feature_scaler = preprocess_data(data)
X_train, X_test, y_train, y_test, y_train_scaled, y_test_scaled, dates_train, dates_test, price_scaler = split_data(data, dates)

torch.save(price_scaler, "price_scaler.pt")
torch.save(feature_scaler, "feature_scaler.pt")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train_scaled.shape)
print("y_test shape:", y_test_scaled.shape)
print("y_train_scaled shape:", y_train_scaled.shape)
print("y_test_scaled shape:", y_test_scaled.shape)
print("dates_train shape:", dates_train.shape)
print("dates_test shape:", dates_test.shape)

