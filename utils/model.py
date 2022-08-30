# lstm & GRU implements
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab_size=None, embedding_dim=None, hidden_size=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_size,
                            num_layers = self.num_layers)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size*2, 1)
    
    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
    def forward(self, inputs, batch_size):
        embeds = self.embeddings(inputs)
        embeds = embeds.view(-1, embeds.shape[-2], embeds.shape[-1])
        output, _ = self.lstm(embeds)

        out_forward = output#[:, -1, :]
        # out_forward = self.fc(embeds)
        text_feature = self.drop(out_forward)

        out = self.fc2(text_feature)
        out = torch.squeeze(out, -1)
        out = self.classifier(out)
        out = torch.squeeze(out, -1)
        out = torch.sigmoid(out)
        return out
    