import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Function to get the list of S&P 500 symbols
def get_sp500_symbols():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    symbols = []

    for row in table.findAll('tr')[1:]:
        symbol = row.findAll('td')[0].text.strip()
        symbols.append(symbol)

    return symbols

# Functions to calculate RSI and Momentum
def rsi(data, period=14):
    delta = data.diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def momentum(data, period=14):
    return data.diff(period)

# Create the NYSE directory if it doesn't exist
if not os.path.exists('NYSE'):
    os.makedirs('NYSE')

# Download and save the data
symbols = get_sp500_symbols()
start_date = '2000-01-01'
end_date = '2023-03-31'

for symbol in tqdm(symbols, desc="Downloading stocks"):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)

        # print year being processed
        current_year = data.index[-1].year
        print(f'Downloading {symbol} ({current_year})')

        data['RSI'] = rsi(data['Close'])
        data['Momentum'] = momentum(data['Close'])
        data.to_csv(f'NYSE/{symbol}.csv')
    except Exception as e:
        print(f'Error downloading {symbol}: {e}')

