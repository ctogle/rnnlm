from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import time
import math

import pdb

class lm(nn.Module):

    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.projection.weight.data.uniform_(-init_range, init_range)
        self.projection.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        wargs = (self.n_layers, batch_size, self.d_hidden)
        if self.rnn_type == 'LSTM':
            hidden = (Variable(weight.new(*wargs).zero_()), 
                      Variable(weight.new(*wargs).zero_()))
        elif self.rnn_type == 'GRU':
            hidden = Variable(weight.new(*wargs).zero_())
        else:
            raise ValueError('unsupported rnn_type: "%s"' % self.rnn_type)
        return hidden

    @staticmethod
    def repackage_hidden(h):
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(lm.repackage_hidden(v) for v in h)

    def __init__(self, n_vocab, d_embed, rnn_type, d_hidden, n_layers, p_drop):
        super(lm, self).__init__()
        self.n_vocab = n_vocab
        self.d_embed = d_embed
        self.rnn_type = rnn_type
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.p_drop = p_drop

        self.encoder = nn.Embedding(self.n_vocab, self.d_embed)
        if self.rnn_type in ('LSTM', 'GRU'):
            rnn_cls = getattr(nn, self.rnn_type)
            self.rnn = rnn_cls(self.d_embed, self.d_hidden, self.n_layers, 
                               dropout=self.p_drop)
        self.projection = nn.Linear(self.d_hidden, self.n_vocab)
        self.decoder = nn.LogSoftmax(dim=1)

        self.init_weights()
        self.epochs = 0

    def forward(self, i, h):
        #          i : (bptt, batch_size)
        e = self.encoder(i)
        #   embedded : (bptt, batch_size, d_embed)
        o, h = self.rnn(e, h)
        #     output : (bptt, batch_size, d_hidden)
        #     hidden : ((n_layers, batch_size, d_hidden), 
        #               (n_layers, batch_size, d_hidden))
        p = self.projection(o.view(o.size(0) * o.size(1), o.size(2)))
        # projection : (bptt * batch_size, n_vocab)
        d = self.decoder(p).view(o.size(0), o.size(1), p.size(1))
        #    decoded : (bptt, batch_size, n_vocab)
        return d, h

    def score(self, sentence):#, n=35):
        self.eval()
        log_p = 0
        words = []
        for w in sentence:
            words.append(w)
            #if len(words) > n:
            #    words.pop(0)
            i = Variable(torch.LongTensor([words]).t().cuda(), volatile=True)
            h = self.init_hidden(1)
            o, h = self(i, h)
            log_p += o.data[-1, 0, w]

            #print('i', i.size())
            #print('o', o.size())
            #print('logp', log_p, o.data[-1, 0, w])

            #log_p += o.data[0, 0, w]
        #pdb.set_trace()
        return log_p * 0.2





    def epoch(self, opt, criterion, loader, log_interval, bptt, batch_size=16):
        self.train()
        self.epochs += 1
        total_loss = 0
        start_time = time.time()
        lr = opt.param_groups[0]['lr']
        h = self.init_hidden(batch_size)
        for t, (i, g) in enumerate(loader.stream(bptt, batch_size)):
            #loader.translate(i, g)
            o, h = self(i, self.repackage_hidden(h))
            loss = criterion(o.view(-1, self.n_vocab), g)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(self.parameters(), 0.25)
            opt.step()
            total_loss += loss.data
            if t % log_interval == 0 and t > 0:
                cur_loss = total_loss[0] / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:5.2f} '
                      '| ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                    self.epochs, t, int(len(loader) / (bptt * batch_size)), lr,
                    elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate(self, criterion, loader, bptt, batch_size):
        self.eval()
        total_loss = 0
        h = self.init_hidden(batch_size)
        for t, (i, g) in enumerate(loader.stream(bptt, batch_size, evaluation=True)):
            o, h = self(i, self.repackage_hidden(h))
            total_loss += len(i) * criterion(o.view(-1, self.n_vocab), g).data
        return total_loss[0] / (loader.batch_count * batch_size)

    def train_model(self, path, train, valid=None, 
                    epochs=10, batch_size=16, bptt=35, lr=0.01, log_interval=50):
        msg = '| end epoch {:3d} | time {:5.2f}s | val-loss {:5.2f} | val-ppl {:8.2f} |'
        opt = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        lr_sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda x: 0.9 ** x])
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.NLLLoss()
        best_val_loss = None
        try:
            print('train model:\n', self)
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                self.epoch(opt, criterion, train, log_interval, bptt, batch_size)
                lr_sched.step(epoch)
                if valid:
                    val_criterion = nn.NLLLoss()
                    val_loss = self.evaluate(val_criterion, valid, bptt, batch_size)
                    val_ppl = math.exp(val_loss)
                    elapsed = (time.time() - epoch_start_time)
                    print('-' * 89)
                    print(msg.format(self.epochs, elapsed, val_loss, val_ppl))
                    print('-' * 89)
                    if not best_val_loss or val_loss < best_val_loss:
                        with open(path, 'wb') as f:
                            torch.save(self, f)
                        best_val_loss = val_loss
                    #else
                    #    # Anneal the learning rate if no improvement has been seen in the validation dataset
                    #    lr /= 4.0
                else:
                    with open(path, 'wb') as f:
                        torch.save(self, f)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')





