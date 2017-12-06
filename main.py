import torch
import torch.nn as nn
from torch.autograd import Variable
import model
import time
import math
import sys
import os
import data

import pdb

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def train_model(model, loader, log_interval, bptt, lr=0.01):
    print('train model', model)
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(loader.vocabulary)
    h = model.init_hidden(loader.batch_size)
    for t, (i, g) in enumerate(loader.stream(bptt)):
        #train.translate(i, g)

        h = repackage_hidden(h)
        #model.zero_grad()
        o, h = model(i, h)

        loss = criterion(o.view(-1, ntokens), g)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
        optimizer.step()

        total_loss += loss.data

        if t % log_interval == 0 and t > 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                model.epoch, t, loader.batch_count, learning_rate,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    model.epoch += 1


if __name__ == '__main__':
    train_path = sys.argv[1]
    batch_size = 8
    train = data.loader(train_path, batch_size)
    learning_rate = 0.01
    bptt = 35

    n_vocab = len(train.vocabulary)
    model = model.lm(n_vocab, 100, 'LSTM', 400, 2, 0.2)
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), 
        lr=learning_rate, momentum=0.99, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    train_model(model, train, 50, bptt)
