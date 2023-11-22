import argparse
from pathlib import Path
import math
import numpy as np
import itertools
import multiprocessing as mp

from .repr import Tokenizer

def analyze(ca):
    files = sorted(ca.data_dir.glob('**/*events.txt'))
    tokenizer = Tokenizer(ca.vocab_file)
    args = zip(files, itertools.repeat(tokenizer, len(files)))
    with mp.Pool() as pool:
        for result in pool.imap(analyze_song, args):
            pass

def analyze_song(args):
    file, tokenizer = args
    events = file.read_text().splitlines()
    h = H(events)
    gs = GS(events)

    # event distribution
    event_hist = [0] * len(tokenizer.vocab)
    for e in events:
        event_hist[tokenizer.e2i(e)] += 1

    return {
        'H': h,
        'GS': gs,
        'D': event_hist,
    }

def H(events, ngram=4):
    bars = events_to_bars(events)
    out = []
    for i in range(len(bars) - ngram + 1):
        nbars = bars[i: i + ngram]
        pc_hist = [0] * 12
        for bar in nbars:
            for e in bar:
                if e.startswith('pitch'):
                    pitch = int(e.split('_')[1])
                    pc = pitch % 12
                    pc_hist[pc] += 1
        pc_hist = [x + 1e-6 for x in pc_hist]
        prob = [x / sum(pc_hist) for x in pc_hist]
        entropy = -sum([p * math.log(p, 2) for p in prob])
        out.append(entropy)
    out = np.mean(out)
    return out

def GS(events):
    bars = events_to_bars(events)
    g_vecs = [] # grooving vector
    out = []
    for bar in bars:
        pos = 0
        v = [0] * 16
        for e in bar:
            if e.startswith('subbeat'):
                pos = int(e.split('_')[1])
            if e.startswith('pitch'):
                v[pos] = 1
        if sum(v) > 0:
            g_vecs.append(v)
    for pair in itertools.combinations(list(range(len(g_vecs))), 2):
        i, j = pair
        xor_sum = sum([g_vecs[i][k] ^ g_vecs[j][k] for k in range(16)])
        gs = 1 - xor_sum / 16
        out.append(gs)
    out = np.mean(out)
    return out

def events_to_bars(events):
    bars = []
    bar = []
    for e in events:
        if e == 'bar' and len(bar) > 0:
            if bar[0] == 'bar':
                bars.append(bar)
            bar = []
        bar.append(e)
    if len(bar) > 0:
        bars.append(bar)
    return bars


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--result_dir', type=Path, required=True)
    parser.add_argument('--vocab_file', type=Path, required=True)
    ca = parser.parse_args()
    analyze(ca)
