import torch
import torch.nn as nn
import model
import data
import math
import time
import os
import argparse

import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train or use an rnn lm')
    parser.add_argument('--model_path', type=str, default='./rnn.lm')
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--valid_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--d_embed', type=int, default=500)
    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--d_hidden', type=int, default=500)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--p_dropout', type=float, default=0.2)
    parser.add_argument('--bptt', type=int, default=35)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=int, default=0.01)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.train_path:
        train = data.loader(args.train_path)
        n_vocab = len(train.vocabulary)

    if args.resume:
        with open(args.model_path, 'rb') as f:
            model = torch.load(f)
    else:
        model = model.lm(n_vocab, args.d_embed, args.rnn_type, 
                         args.d_hidden, args.n_layers, args.p_dropout)
    if args.cuda:
        model.cuda()

    if args.train_path:
        valid = data.loader(args.valid_path) if args.valid_path else None
        model.train_model(args.model_path, train, valid, args.epochs, 
                          args.batch_size, args.bptt, args.learning_rate)
        with open(args.model_path, 'rb') as f:
            model = torch.load(f)

    if args.test_path:
        test = data.loader(args.test_path)
        criterion = nn.CrossEntropyLoss()
        test_loss = model.evaluate(criterion, test, args.bptt, args.batch_size)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
              test_loss, math.exp(test_loss)))
        print('=' * 89)

    if args.input_path:
        input = data.loader(args.input_path)
        pdb.set_trace()

