#!/usr/bin/env python
import contextlib
import tqdm
import lzma
import sys
import os
import argparse

import pdb

'''
prepare raw xz data for training
'''

def xz_stream(path):
    with open(path, 'rb') as f:
        with lzma.LZMAFile(f, 'rb') as decompressed:
            for c, l in enumerate(decompressed):
                try:
                    yield l.decode('utf-8').strip()
                except UnicodeDecodeError:
                    continue

def txt_stream(path):
    with open(path, 'r') as f:
        for c, l in enumerate(f):
            yield l.strip()

@contextlib.contextmanager
def stream(path):
    def stream():
        if path.endswith('.xz'):
            yield from xz_stream(path)
        elif path.endswith('.txt'):
            yield from txt_stream(path)
        else:
            raise ValueError('unsupported input: "%s"' % path)
    yield stream()

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ,. '
def clean(line):
    line = line.upper().replace(',', ' ,').replace('.', ' .')
    for c in line:
        if not c in alphabet:
            break
    else:
        if line.count(' ') > 8:
            yield line

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train or use an rnn lm')
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--max_count', type=int, default=10000)
    args = parser.parse_args()

    with open(args.output_path, 'w') as oh:
        with tqdm.tqdm(total=args.max_count, desc=args.input_path) as pbar:
            with stream(args.input_path) as ih:
                count = 0
                for l in ih:
                    for s in clean(l):
                        oh.write(s)
                        oh.write(os.linesep)
                        count += 1
                        pbar.update(1)
                    if count >= args.max_count:
                        break
