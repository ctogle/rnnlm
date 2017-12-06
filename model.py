from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pdb

class lm(nn.Module):

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        wargs = (self.n_layers, batch_size, self.n_hidden)
        if self.rnn_type == 'LSTM':
            hidden = (Variable(weight.new(*wargs).zero_()), 
                      Variable(weight.new(*wargs).zero_()))
        elif self.rnn_type == 'GRU':
            hidden = Variable(weight.new(*wargs).zero_())
        else:
            raise ValueError('unsupported rnn_type: "%s"' % self.rnn_type)
        return hidden

    def __init__(self, n_vocab, d_embed, rnn_type, n_hidden, n_layers, p_drop):
        super(lm, self).__init__()
        self.n_vocab = n_vocab
        self.d_embed = d_embed
        self.rnn_type = rnn_type
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.p_drop = p_drop
        self.drop = nn.Dropout(self.p_drop)
        self.encoder = nn.Embedding(self.n_vocab, self.d_embed)
        if self.rnn_type in ('LSTM', 'GRU'):
            rnn_cls = getattr(nn, self.rnn_type)
            self.rnn = rnn_cls(self.d_embed, self.n_hidden, self.n_layers, 
                               dropout=self.p_drop)
        self.decoder = nn.Linear(self.n_hidden, self.n_vocab)
        self.init_weights()
        self.epoch = 0

    def forward(self, i, hidden):
        embedded = self.drop(self.encoder(i))
        output, hidden = self.rnn(embedded, hidden)
        output = self.drop(output)
        osize = output.size(0) * output.size(1)
        decoded = self.decoder(output.view(osize, output.size(2)))
        return decoded.view(osize, decoded.size(1)), hidden

