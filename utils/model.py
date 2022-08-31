# Implementations
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class Naive_LSTM(nn.Module):
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

class LSTM(nn.Module):
    def __init__(self, vocab_size=None, embedding_dim=None, hidden_size=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(input_size = embedding_dim,
                            hidden_size = hidden_size,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def create_mask(self, src):
        mask = (src != 0)
        # print(mask.shape)
        return mask

    def forward(self, inputs, batch_size):
        embeds = self.embeddings(inputs)
        embeds = embeds.view(-1, embeds.shape[-2], embeds.shape[-1])
        # output, hidden = self.lstm(embeds)

        output, hidden = self.rnn(embeds)
        output = self.drop(output)
        forward_h = hidden[-2,:,:].contiguous()
        backward_h = hidden[-1,:,:].contiguous()
        hidden = torch.tanh(self.fc(torch.cat((forward_h, backward_h), dim = 1)))
        
        # simple attentioon implementation
        out = torch.tanh(self.fc(output))
        attn = self.v(out).squeeze(2)
        attn = F.softmax(attn, dim = 1)

        attn = attn.unsqueeze(1)
        output = output.permute(1, 0, 2)
        weighted = torch.bmm(attn, out)
        weighted = weighted.permute(1, 0, 2)

        out = self.classifier(weighted)
        out = torch.squeeze(out, -1)
        # out = self.classifier2(weighted)
        # out = torch.squeeze(out, -1)
        out = torch.sigmoid(out[0])
        return out
