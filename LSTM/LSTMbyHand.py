import torch
import torch.nn as nn
import pytorch_lightning as L

class LSTMbyHand(L.LightningModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Initialize weight and biases
        # For forget, input cell and output gates
        # all of this is explained more in the LSTM.pdf
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_forget = nn.Parameter(torch.randn(self.hidden_size, self.input_size), requires_grad=True)
        self.weight_input = nn.Parameter(torch.randn(self.hidden_size, self.input_size), requires_grad=True)
        self.weight_cell = nn.Parameter(torch.randn(self.hidden_size, self.input_size), requires_grad=True)
        self.weight_output = nn.Parameter(torch.randn(self.hidden_size, self.input_size), requires_grad=True)

        self.bias_forget = nn.Parameter(torch.randn(self.hidden_size, 1), requires_grad=True)
        self.bias_input = nn.Parameter(torch.randn(self.hidden_size, 1), requires_grad=True)
        self.bias_cell = nn.Parameter(torch.randn(self.hidden_size, 1), requires_grad=True)
        self.bias_output = nn.Parameter(torch.randn(self.hidden_size, 1), requires_grad=True)

    def forward(self, input):
        # Initialize the cell state and hidden state to 0
        hidden = torch.zeros(self.hidden_size, 1)
        cell = torch.zeros(self.hidden_size, 1)

        for i in range(input.size(0)):
            x = input[i].view(self.input_size, 1)

            forget_gate = torch.sigmoid(self.weight_forget @ x + self.bias_forget)
            input_gate = torch.sigmoid(self.weight_input @ x + self.bias_input)
            c_tilde = torch.tanh(self.weight_cell @ x + self.bias_cell)

            cell = forget_gate * cell + input_gate * c_tilde
            output = torch.sigmoid(self.weight_output @ x + self.bias_output)
            hidden = output * torch.tanh(cell)

        # Return final hidden state
        return hidden

    def configure_optimizers(self):
        # Use the Adam optimizer for training
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        # Unpack the batch into input sequences and labels
        input, label = batch
        # Apply the model to the input sequence to get a prediction
        output = self.forward(input)
        # Compute the squared difference between the prediction and the label as the loss
        loss = (output - label)**2
        # Log the loss
        print(loss.shape)
        self.log("train_loss", loss.mean())
        return loss

