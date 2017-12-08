from torch.autograd import Variable
import torch
import collections
import tqdm

import pdb

class vocabulary:

    def __len__(self):
        return len(self.itot)

    def __getitem__(self, token):
        if token in self.ttoi:
            return self.ttoi[token]
        else:
            return self.unk

    def __init__(self, unk, eos):
        self.ttoi = {}
        self.itot = []
        self.unk = self.add(unk)
        self.eos = self.add(eos)

    def add(self, token):
        if not token in self.ttoi:
            self.ttoi[token] = len(self.itot)
            self.itot.append(token)
        return self.ttoi[token]

class corpus:

    def read(self, lc=None):
        lc = self.linecount if lc is None else lc
        with tqdm.tqdm(total=lc, desc=self.path) as pbar:
            with open(self.path, 'r') as f:
                for j, l in enumerate(f):
                    yield l
                    pbar.update(1)

    def __len__(self):
        return self.tokens
    
    def __init__(self, path, vocab=None, lc=None, min_count=10):
        self.path = path
        self.tokens = 0
        self.linecount = 0

        if not vocab is None:
            for line in self.read(lc):
                words = line.split()
                self.tokens += len(words) + 1
                self.linecount += 1
            self.vocabulary = vocab
        else:
            counter = collections.Counter()
            for line in self.read(lc):
                words = line.split()
                counter.update(words)
                self.tokens += len(words) + 1
                self.linecount += 1

            self.vocabulary = vocabulary('<unk>', '<eos>')
            for token, count in counter.most_common():
                if count < min_count:
                    break
                else:
                    self.vocabulary.add(token)

        info = (self.path, self.linecount, self.tokens, len(self.vocabulary))
        print('corpus "%s" | %d lines | %d tokens | %d token vocabulary' % info)

    def __iter__(self):
        '''yield the tokens of the corpus with eos tokens between lines'''
        for line in self.read():
            for word in line.split():
                yield self.vocabulary[word]
            else:
                yield self.vocabulary.eos

class loader(corpus):

    def translate(self, i, o):
        print('i | ', ' '.join([self.vocabulary.itot[x] for x in i.data[:,0]]))
        print('o | ', ' '.join([self.vocabulary.itot[x] for x in o.data[:]]))

    def stream(self, n, batch_size, evaluation=False, cuda=True):
        source = torch.LongTensor(len(self))
        for t, index in enumerate(self):
            source[t] = index
        self.batch_count = source.size(0) // batch_size
        source = source.narrow(0, 0, self.batch_count * batch_size)
        source = source.view(batch_size, -1).t().contiguous()
        if cuda:
            source = source.cuda()
        for t in range(0, source.size(0) - 1, n):
            seq_len = min(n, len(source) - 1 - t)
            i = Variable(source[t:t + seq_len], volatile=evaluation)
            o = Variable(source[t + 1:t + 1 + seq_len].view(-1))
            yield i, o
