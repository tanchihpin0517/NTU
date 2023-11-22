from copy import deepcopy
from random import randint
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import functools
import numpy as np
import argparse
import miditoolkit

from .repr import Tokenizer
from . import utils

class AILabs1k7Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer,
        use_cache=True,
        n_songs=None,
    ):
        self.songs = self.load_data(
            data_dir,
            tokenizer,
            use_cache=use_cache,
            n_songs=n_songs,
        )

        len_events = [len(song['events']) for song in self.songs]

        print('[mean and std]')
        print(f'\torig music data: {np.mean(len_events)}, {np.std(len_events)}')

    def load_data(
        self,
        data_dir,
        tokenizer,
        use_cache=True,
        n_songs=None
    ):
        dir_name = data_dir.name
        cache_file = data_dir.parent / f'{dir_name}_cache_{n_songs if n_songs else "full"}_bd_{tokenizer.beat_div}.pkl'
        if use_cache:
            if cache_file.exists():
                print(f'loading cached data from {cache_file}')
                return utils.pickle_load(cache_file)

        midi_files = list((data_dir / "midi_analyzed").glob('**/*.mid'))
        midi_files.sort(key = lambda x: int(Path(x).stem.split('_')[0]))

        if n_songs is not None:
            midi_files = midi_files[:n_songs]

        map_args = list(zip(midi_files, [tokenizer]*len(midi_files)))
        cache_out_dir = data_dir.parent / f"{dir_name}_cache_bd_{tokenizer.beat_div}"
        cache_out_dir.mkdir(exist_ok=True)
        songs = []

        with mp.Pool() as pool:
            for i, song in enumerate(tqdm(pool.imap(self.load_data_map, map_args), total=len(map_args))):
                songs.append(song)

                song_dir = cache_out_dir / f'{i}'
                song_dir.mkdir(exist_ok=True)

                utils.pickle_save(song, (song_dir / 'song.pkl'))

                shutil.copy(song['source'], song_dir / 'source.mid')
                (song_dir / 'meta.txt').write_text(str(song['metadata']))

                (song_dir / 'events.txt').write_text("\n".join(map(str, song['events'])))
                tokenizer.events_to_midi(
                    song['events'],
                    song['metadata']['beat_per_bar']
                ).dump(song_dir / 'events.mid')

        utils.pickle_save(songs, cache_file)

        return songs

    @staticmethod
    def load_data_map(args):
        midi_file, tokenizer = args

        midi_objs = miditoolkit.midi.parser.MidiFile(midi_file)
        song = tokenizer.get_song_from_midi(midi_objs)
        song['source'] = midi_file

        return song

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        return self.songs[idx]

    @classmethod
    def get_dataloader(cls, tokenizer, max_seq_len, *args, **kwargs):
        return DataLoader(
            *args,
            **kwargs,
            collate_fn=functools.partial(
                cls.collate_fn,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            ),
        )

    @staticmethod
    def collate_fn(batch, tokenizer, max_seq_len):
        song_segs = []
        label_segs = []

        song_lens = []

        for song in batch:
            # events = deepcopy(song['events'])
            ids = deepcopy(song['ids'])

            idx_from = randint(0, len(ids) // (max_seq_len // 2)) * (max_seq_len // 2)
            idx_to = min(idx_from + max_seq_len, len(ids))
            song_seg = ids[idx_from:idx_to]

            # randomly mask
            pass

            # transpose
            # TODO: chord
            # pitches = [e.pitch for e in events if isinstance(e, NoteEvent)]
            # pitches.sort()
            # pitch_range = [min(pitches), max(pitches)]
            # shift_range = [21-pitch_range[0], 108-pitch_range[1]]
            # shift = randrange(*shift_range)
            #
            # for s in [song_seg, ls_seg, label_seg]:
            #     for i, event in enumerate(s):
            #         if isinstance(event, NoteEvent):
            #             s[i].pitch += shift
            #             assert s[i].pitch >= 21 and s[i].pitch < 109, s[i]

            # add bos
            bos_id = tokenizer.e2i('spec_bos')
            song_seg = [bos_id] + song_seg

            # to tensor
            song_seg = torch.LongTensor(song_seg)
            label_seg = torch.LongTensor(song_seg)

            song_lens.append(len(song_seg))

            # if not start from beginning, only consider the second half part
            # if song_from != 0:
            #     label_seg[:len(label_seg)//4] = -100

            # pad & truncate
            song_seg = F.pad(song_seg, (0, max_seq_len-len(song_seg)), 'constant', 0)
            label_seg = F.pad(label_seg, (0, max_seq_len-len(label_seg)), 'constant', 0)

            # song_seg = song_seg[:max_seq_len]
            assert song_seg.shape == label_seg.shape
            # assert ls_seg.shape[0] == ls_align.shape[0]

            # label ignore
            label_seg[label_seg == 0] = -100
            # label_seg[:, 0][(label_seg[:, 0] == -100) * (label_seg[:, 1] != -100)] = 0 # family 0

            # append
            song_segs.append(song_seg)
            label_segs.append(label_seg)

        song_segs = torch.stack(song_segs, dim=0)
        label_segs = torch.stack(label_segs, dim=0)

        song_segs = song_segs[:, :max(song_lens)]
        label_segs = label_segs[:, :max(song_lens)]

        return {
            'source': [song['source'] for song in batch],
            'metadata': [song['metadata'] for song in batch],
            'song_ids': song_segs,
            'label_ids': label_segs,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cmd_cache = subparsers.add_parser('gen_cache')
    cmd_cache.add_argument('--data_dir', type=Path, required=True)
    cmd_cache.add_argument('--vocab_file', type=Path, required=True)
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        exit()
    elif args.command == 'gen_cache':
        dataset = AILabs1k7Dataset(
            args.data_dir,
            tokenizer=Tokenizer(args.vocab_file, beat_div = 4, ticks_per_beat = 480),
            use_cache=False,
        )

