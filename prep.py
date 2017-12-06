import contextlib
import tqdm
import lzma
import sys
import os

import pdb

'''
prepare raw xz data for training
'''

@contextlib.contextmanager
def xz_stream(path):
    def stream():
        with open(path, 'rb') as f:
            with lzma.LZMAFile(f, 'rb') as decompressed:
                for c, l in enumerate(decompressed):
                    try:
                        yield l.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        continue
    yield stream()

def clean(line):
    yield line.upper()

if __name__ == '__main__':
    raw = sys.argv[1]
    prepped = sys.argv[2]
    max_count = int(sys.argv[3])

    with open(prepped, 'w') as oh:
        with tqdm.tqdm(total=max_count, desc=raw) as pbar:
            with xz_stream(raw) as ih:
                count = 0
                for l in ih:
                    for s in clean(l):
                        oh.write(s)
                        oh.write(os.linesep)
                        count += 1
                        pbar.update(1)
                    if count >= max_count:
                        break
