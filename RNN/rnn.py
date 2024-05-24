'''
Recurrent Neural Network
single layer and hidden size of 128
applies multi-layer Elman RNN with tanh or ReLU non-linearity to an input 
sequence.
'''

import torch.nn as nn

class RNN(nn.Module):
    # Define the RNN modules' constructor
    def __init__(self, input_size, hidden_size, output_size, num_layers=10):
        # call constuctor of parent class (nn.Module)
        super(RNN, self).__init__()
        # save the hidden size for later use
        self.hidden_size = hidden_size
        # Define the RNN layer with the specified input size, hidden size, and 
        # number of layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        # Define fully connected layer to map from the hidden size to the 
        # output size
        self.fc = nn.Linear(hidden_size, output_size)

    # Define the forward function, which takes input tensor `x` and the hidden 
    # state tensor `h`
    def forward(self, x, h):
        h = h.to(x.device)
        # Apply the RNN layer to the input tensor `x` and hidden state tensor `h`
        out, h = self.rnn(x,h)
        # Apply fully connected layer to the output of the RNN layer at the 
        # final time step
        out = self.fc(out[:, -1, :])
        # Return the output and the final hidden state tensor `h`
        return out, h
