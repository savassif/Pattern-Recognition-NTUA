import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from time import time 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        self.lengths = [len(feat) for feat in feats] # Find the lengths
        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype('int64')
    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        get_max=max([len(subar) for subar in x])
        feut_dim=len(x[0][0])
        padded =np.zeros((len(x),get_max,feut_dim))
        # --------------- Insert your code here ---------------- #
        for i in range(len(x)):
            for j in range(len(x[i])):
                for k in range(feut_dim):
                    padded[i][j][k]=x[i][j][k]
        return padded
    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]
    def __len__(self):
        return len(self.feats)

class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False,dropout = 0,pad=False,DEVICE=torch.device('cpu')):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.layers = num_layers
        self.hidden_size = rnn_size
        self.lstm = nn.LSTM(input_size=input_dim,hidden_size=rnn_size,num_layers=num_layers,dropout=dropout,bidirectional=bidirectional,batch_first=True)
        self.clf = nn.Linear(in_features=self.feature_size,out_features=output_dim)
        self.pad=pad
        self.DEVICE =DEVICE
        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
    def forward(self, x, lengths):
        """
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index
            lengths: N x 1
         """
        # Initial Hidden State h_0
        h_0 = torch.zeros(self.layers*(1+self.bidirectional),len(x),self.hidden_size).double().to(self.DEVICE)
        # Initial Cell State c_0 (same as h_0)
        c_0 = torch.zeros(self.layers*(1+self.bidirectional),len(x),self.hidden_size).double().to(self.DEVICE)
        if self.pad:
            x = pack_padded_sequence(x.to(self.DEVICE), lengths, batch_first=True).to(self.DEVICE)
        lstm_out,(h_n,c_n) = self.lstm(x.to(self.DEVICE),(h_0,c_0))
        if self.pad:
            lstm_out,_=pad_packed_sequence(lstm_out,batch_first=True)
        last_outputs = self.clf(self.last_timestep(lstm_out,lengths,self.DEVICE,self.bidirectional))
        return last_outputs
    def last_timestep(self, outputs, lengths,DEVICE, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths,DEVICE)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)
        else:
            return self.last_by_index(outputs, lengths,DEVICE)
    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward
    @staticmethod
    def last_by_index(outputs, lengths,DEVICE):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, (idx.long()).to(DEVICE)).squeeze()

    