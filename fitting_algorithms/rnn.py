"""
Recurrent neural network implementation.
"""

__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve'
__email__ = "saul.alonso.monsalve@cern.ch"

import torch
import torch.nn as nn
from torch.autograd import Variable


class FittingRNN(nn.Module):
    def __init__(self,
                 rnn: str,  # type of RNN {"lstm", "gru"}
                 nb_layers: int,  # number of layers
                 sum_outputs: bool,  # whether to sum outputs from RNN layers
                 nb_lstm_units: int,  # units of each LSTM/GRU layer
                 input_size: int,  # size of each item in the input sequence
                 output_size: int,  # size of each item in the output sequence
                 batch_size: int,  # batch size
                 dropout: float,  # dropout value
                 bidirectional: bool,  # whether to have a bi-directional RNN
                 learn_init_states: bool,  # whether to learn the initial states
                 init_states: str,  # initial state initialization {"rand", other}
                 device: object  # device
                 ):
        super(FittingRNN, self).__init__()

        self.hidden = None
        self.rnn_type = rnn
        if self.rnn_type == "lstm":
            self.rnn_model = nn.LSTM
        else:
            self.rnn_model = nn.GRU
        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.sum_outputs = sum_outputs
        self.input_size = input_size
        self.batch_size = batch_size
        self.nb_out = output_size
        self.bidirectional = bidirectional
        self.learn_init_states = learn_init_states
        self.dropout = dropout
        self.device = device

        if "rand" in init_states:
            self.init_states = torch.randn
        else:
            self.init_states = torch.zeros

        # when the model is bidirectional we double the output dimension
        self.D = 2 if self.bidirectional == True else 1

        # build actual NN
        self.__build_model()

    def __build_model(self):
        self.proj_input = nn.Linear(self.input_size, self.nb_lstm_units)
        self.activation = nn.ReLU()

        if self.sum_outputs:
            # design RNN
            self.rnn = []
            for layer in range(self.nb_layers):
                input_size = self.D * self.nb_lstm_units
                if layer == 0:
                    input_size = self.nb_lstm_units
                self.rnn.append(self.rnn_model(input_size=input_size, hidden_size=self.nb_lstm_units,
                                               num_layers=1, batch_first=True, dropout=0.0,
                                               bidirectional=self.bidirectional,
                                               )
                                )
            self.rnn = nn.ModuleList(self.rnn)
            self.dropout_layer = nn.Dropout(self.dropout, inplace=True)
        else:
            input_size = self.nb_lstm_units
            self.rnn = self.rnn_model(input_size=input_size, hidden_size=self.nb_lstm_units,
                                      num_layers=self.nb_layers, batch_first=True, dropout=self.dropout,
                                      bidirectional=self.bidirectional,
                                      )

        # output layer which projects back to output space
        self.hidden_to_out = nn.Linear(self.D * self.nb_lstm_units, self.nb_out)

        if self.learn_init_states:
            self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for layer in range(self.nb_layers if self.sum_outputs else 1):
            # initial hidden states
            if self.sum_outputs:
                # case 1: we stack the rnn layer outputs manually
                if self.learn_init_states:
                    hidden_h = self.init_states(self.D, 1, self.nb_lstm_units)
                else:
                    hidden_h = self.init_states(self.D, self.batch_size, self.nb_lstm_units)
            else:
                # case 2: we use a RNN model with nb_layers
                hidden_h = self.init_states(self.D * self.nb_layers, self.batch_size, self.nb_lstm_units)
            if not self.learn_init_states:
                # send initial hidden states to device
                hidden_h = hidden_h.to(self.device)
            if self.rnn_type == "lstm":
                if self.sum_outputs:
                    # case 1: we stack the rnn layer outputs manually
                    if self.learn_init_states:
                        hidden_c = self.init_states(self.D, 1, self.nb_lstm_units)
                    else:
                        hidden_c = self.init_states(self.D, self.batch_size, self.nb_lstm_units)
                else:
                    # case 2: we use a RNN model with nb_layers
                    hidden_c = self.init_states(self.D * self.nb_layers, self.batch_size, self.nb_lstm_units)
                if not self.learn_init_states:
                    # send initial cell states to device
                    hidden_c = hidden_c.to(device)
            if self.learn_init_states:
                # if we want to learn the initial hidden [and cell] states -> create trainable parameters
                hidden_h = nn.Parameter(hidden_h, requires_grad=True)
                if self.rnn_type == "lstm":
                    # create trainable parameters for initial cell states too
                    hidden_c = nn.Parameter(hidden_c, requires_grad=True)
                    hidden_s = (hidden_h, hidden_c)
            else:
                # if we don't want to learn the initial hidden [and cell] states -> create variable
                hidden_h = Variable(hidden_h)
                if self.rnn_type == "lstm":
                    hidden_c = Variable(hidden_c)
                    hidden_s = (hidden_h, hidden_c)

            if self.rnn_type != "lstm":
                # initial hidden inputs consist of initial hidden states only for GRU 
                hidden_s = hidden_h
            if self.sum_outputs:
                # append the initial hidden and cell states for each layer
                if self.rnn_type == "lstm":
                    hidden.extend(hidden_s)
                else:
                    hidden.append(hidden_s)
            else:
                hidden = hidden_s
        if self.learn_init_states:
            if self.sum_outputs or (not self.sum_outputs and self.rnn_type == "lstm"):
                # pack initial hidden [and cell] states into parameter lists
                hidden = nn.ParameterList([state for state in hidden])
        return hidden

    def forward(self, x_in, x_lengths):
        if not self.learn_init_states:
            # reset the LSTM hidden state if the initial hidden [and cell] parameters are not learnt.
            self.hidden = self.init_hidden()

        x = x_in
        batch_size, seq_len, n_feats = x.shape

        # preprocess the input with a feed-forward layer
        x = x.contiguous()
        x = x.view(-1, n_feats)
        x = self.proj_input(x)
        x = self.activation(x)
        x = x.view(batch_size, seq_len, self.nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)

        out_rnn = 0.
        for layer in range(self.nb_layers if self.sum_outputs else 1):
            if self.sum_outputs:
                if self.rnn_type == "lstm":
                    hidden = (self.hidden[layer * 2], self.hidden[layer * 2 + 1])
                else:
                    hidden = self.hidden[layer]
            else:
                hidden = self.hidden

            if self.learn_init_states:
                # share initial hidden [and cell] states for each element of the batch
                if self.rnn_type == "lstm":
                    hidden = (hidden[0].repeat(1, batch_size, 1), hidden[1].repeat(1, batch_size, 1))
                else:
                    hidden = hidden.repeat(1, batch_size, 1)

            # now run through RNN
            if self.sum_outputs:
                out, hidden = self.rnn[layer](x, hidden)
            else:
                out, hidden = self.rnn(x, hidden)

            # unpack
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

            # dropout after all but last recurrent layer
            if self.sum_outputs and layer < self.nb_layers - 1:
                self.dropout_layer(out)

            # if multiple layers, sum the outputs of all layers instead of using only the last one,
            # similar to a ResNet or DenseNet
            out_rnn += out

            # pack again: input of next RNN layer is output of current RNN layer
            x = torch.nn.utils.rnn.pack_padded_sequence(out, x_lengths, batch_first=True, enforce_sorted=False)

        # reshape and run final linear layer
        out_rnn = out_rnn.contiguous()
        out_rnn = out_rnn.view(-1, out_rnn.shape[2])
        self.dropout_layer(out_rnn)
        out_rnn = self.hidden_to_out(out_rnn)
        out_rnn = out_rnn.view(batch_size, seq_len, self.nb_out)

        # long skip connection (output will be input + small adjustments learnt through the RNN)
        out_rnn += x_in[:, :, :3]  # learn residuals for x,y,z

        return out_rnn
