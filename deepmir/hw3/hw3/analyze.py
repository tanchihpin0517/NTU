import argparse
from pathlib import Path
import math
import numpy as np
import itertools
import multiprocessing as mp
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy

from .repr import Tokenizer

def norm(v, eps=0.0):
    v = [(x + eps) for x in v]
    v = [x / sum(v) for x in v]
    return v


def analyze(ca):
    files = sorted(ca.output_dir.glob('**/*events.txt'))
    tokenizer = Tokenizer(ca.vocab_file)
    args = zip(files, itertools.repeat(tokenizer, len(files)))

    result_file = ca.result_dir / f'{ca.output_dir.name}.json'
    with mp.Pool() as pool:
        results = []
        for result in tqdm(pool.imap(analyze_song, args), total=len(files), desc=result_file.stem):
            results.append(result)
        json.dump(results, result_file.open('w'), indent=2)

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
        'EH': event_hist,
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
        # pc_hist = [x + 1e-6 for x in pc_hist]
        # prob = [x / sum(pc_hist) for x in pc_hist]
        prob = norm(pc_hist, 1e-6)
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

def statistic(ca):
    stats = []
    gt = {}
    for file in sorted(ca.result_dir.glob('**/*.json')):
        if len(ca.filter) > 0 and "dataset" not in str(file) and not any([f in str(file) for f in ca.filter]):
            continue
        song_stats = json.load(file.open())
        tag = file.stem if "dataset" in file.stem else f'{file.parent.stem}_{file.stem}'
        h = []
        gs = []
        ed = [0] * 373
        for ss in song_stats:
            h.append(ss['H'])
            gs.append(ss['GS'])
            for i, x in enumerate(ss['EH']):
                ed[i] += x
        h = np.mean(h)
        gs = np.mean(gs)
        ed = norm(ed)
        stat = {
            'tag': tag,
            'H': h,
            'GS': gs,
            'ED': ed,
        }
        stats.append(stat)
        if "dataset" in tag:
            gt = stat

    gt = deepcopy(gt)
    for stat in stats:
        ed_l2 = l2_dist(gt['ED'], stat['ED']) * 1e6
        stat['ED'] = ed_l2

    if ca.order is not None:
        print(f"sort by {ca.order}")
        stats = sorted(stats, key=lambda x: x[ca.order], reverse=True)

    summary_file = ca.result_dir / f'summary_{"_".join(ca.filter)}.csv'
    out = ["tag,H,GS,ED"]
    for stat in stats:
        # ed_ce = cross_entropy(gt['ED'], stat['ED'])
        # ed_l2 = l2_dist(gt['ED'], stat['ED']) * 1e6
        # print(f"{stat['tag']}: H={stat['H']:.4f}, GS={stat['GS']:.4f}, ED={ed_ce:.4f}")
        out.append(f"{stat['tag']},{stat['H']:.4f},{stat['GS']:.4f},{stat['ED']:.4f}")
    summary_file.write_text('\n'.join(out))

def cross_entropy(p, q):
    eps = 1e-6
    p = norm(p, eps)
    q = norm(q, eps)
    return -sum([p[i] * math.log(q[i], 2) for i in range(len(p))])

def l1_dist(p, q):
    return sum([abs(p[i] - q[i]) for i in range(len(p))]) / len(p)

def l2_dist(p, q):
    return sum([(p[i] - q[i]) ** 2 for i in range(len(p))]) / len(p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    cmd_analyze = subparsers.add_parser('analyze')
    cmd_analyze.add_argument('--output_dir', type=Path, required=True)
    cmd_analyze.add_argument('--result_dir', type=Path, required=True)
    cmd_analyze.add_argument('--vocab_file', type=Path, required=True)

    cmd_stats = subparsers.add_parser('stats')
    cmd_stats.add_argument('--result_dir', type=Path, required=True)
    cmd_stats.add_argument('--order', type=str)
    cmd_stats.add_argument('--filter', nargs='+')

    ca = parser.parse_args()
    if ca.command is None:
        parser.print_help()
        exit()
    elif ca.command == 'analyze':
        analyze(ca)
    elif ca.command == 'stats':
        if ca.filter is None:
            ca.filter = []
        statistic(ca)
    else:
        raise NotImplementedError()
