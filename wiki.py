#!/usr/bin/env python
import argparse
import wikipedia
import os

import pdb

def search(q):
    results = wikipedia.search(q)
    print('results for query: %s' % q)
    for j, r in enumerate(results):
        print('\t%2d | %s' % (j, r))
    return results[0]

def write(path, gen):
    with open(path, 'w') as f:
        for l in gen:
            f.write(l + os.linesep)

def prepare(text):
    for paragraph in text.split(os.linesep):
        for line in paragraph.split('.'):
            yield line

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', default='Python (Programming Language)')
    parser.add_argument('--output', default='wiki.page.txt')
    args = parser.parse_args()

    query = search(args.query)
    page = wikipedia.page(query)
    write(args.output, prepare(page.content))
