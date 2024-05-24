# Machine Learning Research Project

This repository contains the code and resources for an HSI-STEM Machine Learning research project.

## Prerequisites

1. **Docker**: Docker is required to run the application in a containerized environment. If Docker is not already installed on your machine, you can download it from the official Docker website: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).

## Setup and Running the Application

1. Download and install Docker from the provided link if you haven't done so already.

2. Search for the Docker image `aarongc43/rnn-model` in the Docker Desktop application.

3. Pull the Docker image by clicking the "Pull" button associated with the image.

4. Once the image is downloaded, click the "Run" button next to "Pull" to start the model.
   The model will automatically begin running.

The application will now be running inside a Docker container. You can view the output and logs of the application within the Docker Desktop interface.

## Project Files

- `Dockerfile`: Contains the instructions to build the Docker image.
- `requirements.txt`: Lists the Python packages that the application depends on.
- `train.py`: The main script that trains the RNN model.
- `rnn.py`: Contains the definition of the RNN model.
- `visualize.py`: A script for visualizing the model's predictions.
- `data_preprocessing.py`: A script for preprocessing the stock price data.
- `download_data.py`: A script for downloading stock price data.
- `*.npy`: Numpy array files used for training and testing the model.
- `*.pt`: PyTorch model files.

Feel free to explore the code and resources to understand the project in more detail.
