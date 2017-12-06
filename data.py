from torch.autograd import Variable
import torch
import tqdm

import pdb

def batch(it, bsize):
    j, batch = 0, []
    for i in it:
        batch.append(i)
        j += 1
        if j == bsize:
            yield batch
            j, batch = 0, []
    else:
        if batch:
            yield batch

class vocabulary:

    def __len__(self):
        return len(self.itot)

    def __init__(self):
        self.ttoi = {}
        self.itot = []

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

    def __init__(self, path, lc=100000):
        self.path = path
        self.linecount = 0
        self.vocabulary = vocabulary()

        self.tokens = 0
        self.vocabulary.add('<eos>')
        for line in self.read(lc):
            for word in line.split():
                self.tokens += 1
                self.vocabulary.add(word)
            else:
                self.tokens += 1
            self.linecount += 1

    def __iter__(self):
        '''yield the tokens of the corpus with eos tokens between lines'''
        for line in self.read():
            for word in line.split():
                yield self.vocabulary.ttoi[word]
            else:
                yield self.vocabulary.ttoi['<eos>']

class loader(corpus):

    def __init__(self, path, batch_size=8):
        super(loader, self).__init__(path)
        self.batch_size = batch_size
        self.batch_count = self.tokens // self.batch_size

    def translate(self, i, o):
        print('i | ', ' '.join([self.vocabulary.itot[x] for x in i.data[:,0]]))
        print('o | ', ' '.join([self.vocabulary.itot[x] for x in o.data[:]]))

    def stream(self, n, evaluation=False, cuda=True):
        source_length = self.batch_count * self.batch_size
        source = torch.LongTensor(source_length)
        for t, index in enumerate(self):
            if t < source_length:
                source[t] = index
            else:
                break
        source = source.narrow(0, 0, self.batch_count * self.batch_size)
        source = source.view(self.batch_size, -1).t().contiguous()
        if cuda:
            source = source.cuda()

        for t in range(source_length):
            seq_len = min(n, source_length - 1 - t)
            i = Variable(source[t:t + seq_len], volatile=evaluation)
            o = Variable(source[t + 1:t + 1 + seq_len].view(-1))
            yield i, o

