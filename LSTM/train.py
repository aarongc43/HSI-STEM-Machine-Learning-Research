import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as L
from LSTMbyHand import LSTMbyHand
# from LightningLSTM import LightningLSTM

# Create the training data for the neural network
# Each sample in the sequence now has 4 features
input = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([[0.], [1.]])

# Wrap the inputs and labels in a TensorDataset, which provides a convenient way to iterate over the dataset 
dataset = TensorDataset(input, labels)
# Wrap the dataset in a DataLoader, which provides batching, shuffling, etc. 
dataLoader = DataLoader(dataset)

def train_model(model, max_epochs=300):
    # Create pytorch lightning trainer with the specified number of epochs
    trainer = L.Trainer(max_epochs=max_epochs, log_every_n_steps=2)
    # Train the model on the data
    trainer.fit(model)

def print_model_output(model):
    # Print the parameters of the model
    print("After optimization, the parameters are ...")
    for name, param in model.named_parameters():
        print(name, param.data)

    # Print the predictions of the model for the input sequences
    print("\nNow let's compare the observed and predicted values...")
    print("Company A: Observed = 0, Predicted = ", model(torch.tensor([0., 0.5, 0.25, 1.]).unsqueeze(0)).item())
    print("Company B: Observed = 1, Predicted = ", model(torch.tensor([1., 0.5, 0.25, 1.]).unsqueeze(0)).item())

def main():
    # Create an instance of the LSTMbyHand model
    model = LSTMbyHand(input_size=4, hidden_size=1)
    model.train_dataloader = lambda: dataLoader
    # Train model
    train_model(model)
    # Print the model parameters and predictions
    print_model_output(model)

    # Uncomment these lines when LightningLSTM is updated too
    # Create an instance of the LightningLSTM model
    # model = LightningLSTM(input_size=4, hidden_size=1)
    # model.train_dataloader = lambda: dataLoader
    # Train model
    # train_model(model)
    # Print model parameters and predictions
    # print_model_output(model)

if __name__ == "__main__":
    main()

