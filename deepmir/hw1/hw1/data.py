import argparse
import functools
from pathlib import Path
import demucs.separate
import shutil
from torch.utils.data import Dataset, DataLoader
import torchaudio as ta
import multiprocessing as mp
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

from . import utils

SAMPLE_RATE = 16000
SEGMENT_LEN = 5 # seconds

class HW1Dataset(Dataset):
    def __init__(
        self,
        data_file,
        singer_file,
        cache_file=None,
        use_cache=True,
    ):
        self.data = []
        self.singers = singer_file.read_text().splitlines()
        self.singer_ids = {singer: i for i, singer in enumerate(self.singers)}

        if cache_file is None:
            cache_file = data_file.parent / (data_file.stem + '.pkl')

        if use_cache and cache_file.exists():
            print(f"Loading cache from {cache_file} ...")
            self.data = utils.pickle_load(cache_file)
        else:
            with mp.Pool() as pool:
                data_entries = data_file.read_text().splitlines()[:]
                pbar = tqdm(total=len(data_entries), desc="0")
                for result in pool.imap(HW1Dataset.proc_audio, data_entries):
                    for seg in result['segments']:
                        self.data.append({
                            'file': result['file'],
                            'singer': result['singer'],
                            'title': result['title'],
                            'segment': seg,
                        })
                    pbar.set_description(f"{len(self.data)}")
                    pbar.update(1)
                pbar.close()
            utils.pickle_save(self.data, cache_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def proc_audio(entry):
        file, singer, title = entry.split('\t')
        file = Path(file)
        vocal_file = Path(*(file.parts[:-4] + ((file.parts[-4] + "-seperated"),) + file.parts[-3:]))
        vocal_file = vocal_file.parent / (vocal_file.stem + '-vocals.wav')
        wav, sr = ta.load(vocal_file)

        # 1 channel
        wav = wav.mean(0, keepdim=True)
        # resample
        wav = ta.transforms.Resample(sr, SAMPLE_RATE)(wav)
        # split on silence
        segments = split_on_silence(wav, SAMPLE_RATE)

        # format to 5 seconds
        out = []
        seg_exp_size = SAMPLE_RATE * SEGMENT_LEN
        for seg in segments:
            if seg.shape[1] < seg_exp_size:
                t = np.pad(seg, ((0, 0), (0, seg_exp_size - seg.shape[1])))
                out.append(t)
            else:
                for i in range(0, seg.shape[1], seg_exp_size):
                    t = seg[:, i: i + seg_exp_size]
                    if t.shape[1] < seg_exp_size:
                        t = np.pad(t, ((0, 0), (0, seg_exp_size - t.shape[1])))
                    out.append(t)

        return {
            'file': file,
            'singer': singer,
            'title': title,
            'segments': out,
        }

    def get_dataloader(self, *args, **kwargs):
        return DataLoader(
            self,
            *args,
            **kwargs,
            collate_fn = functools.partial(
                HW1Dataset.collate_fn,
                singer_ids = self.singer_ids,
            ),
        )

    @staticmethod
    def collate_fn(batch, singer_ids):
        frames = []
        labels = []

        for item in batch:
            frame = torch.from_numpy(item['segment'])
            singer_id = torch.LongTensor([singer_ids[item['singer']]])

            frames.append(frame)
            labels.append(singer_id)

        frames = torch.stack(frames, dim=0)
        labels = torch.cat(labels, dim=0)

        return {
            "frames": frames,
            "labels": labels,
        }

class HW1TestingDataset(Dataset):
    def __init__(
        self,
        data_dir,
        cache_file=None,
        use_cache=True,
    ):
        self.data = []

        if cache_file is None:
            cache_file = data_dir.parent / (data_dir.stem + '.pkl')

        if use_cache and cache_file.exists():
            print(f"Loading cache from {cache_file} ...")
            self.data = utils.pickle_load(cache_file)
        else:
            with mp.Pool() as pool:
                data_files = sorted(data_dir.glob('*'))[:]
                pbar = tqdm(total=len(data_files), desc="0")
                for result in pool.imap(self.proc_audio, data_files):
                    self.data.append(result)
                    pbar.set_description(f"{len(self.data)}")
                    pbar.update(1)
                pbar.close()
            utils.pickle_save(self.data, cache_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def proc_audio(file):
        vocal_file = file.parent.parent / (file.parent.stem + "-seperated") / (file.stem + '-vocals.wav')
        wav, sr = ta.load(vocal_file)

        # 1 channel
        wav = wav.mean(0, keepdim=True)
        # resample
        wav = ta.transforms.Resample(sr, SAMPLE_RATE)(wav)
        # split on silence
        segments = split_on_silence(wav, SAMPLE_RATE)

        # format to 5 seconds
        out = []
        seg_exp_size = SAMPLE_RATE * SEGMENT_LEN
        for seg in segments:
            if seg.shape[1] < seg_exp_size:
                seg = np.pad(seg, ((0, 0), (0, seg_exp_size - seg.shape[1])))
                out.append(seg)
            else:
                # sliding window with 50% overlap
                half = seg_exp_size // 2
                for i in range(0, (seg.shape[1] - half), half):
                    t = seg[:, i: i + seg_exp_size]
                    if t.shape[1] < seg_exp_size:
                        t = np.pad(t, ((0, 0), (0, seg_exp_size - t.shape[1])))
                    out.append(t)

        if len(out) == 0:
            out.append(np.zeros((1, seg_exp_size)))
            print(f"Warning: no segment found: {file}")

        return {
            'file': file,
            'segments': out,
        }

    def get_dataloader(self, *args, **kwargs):
        return DataLoader(
            self,
            *args,
            **kwargs,
            collate_fn = self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        out = []

        for item in batch:
            tmp = []
            for seg in item['segments']:
                tmp.append(torch.from_numpy(seg))
            song_frames = torch.stack(tmp, dim=0)
            out.append({
                'file': item['file'],
                'frames': song_frames
            })

        return out

def split_on_silence(
    wav,
    sample_rate,
    min_segment_len=1000, # ms
    min_silence_len=1000, # ms
    chunk_len=100, # ms
    threadhold=-20, # db
):
    hop_len = sample_rate // (1000 // chunk_len)
    spec = ta.transforms.MelSpectrogram(n_fft=(hop_len*2))(wav)
    db = ta.transforms.AmplitudeToDB()(spec)
    avg_db = db.mean(1)
    # plot_waveform(avg_db, 1, stem="avg_db")

    silent = avg_db < threadhold
    min_cut_len = min_silence_len // chunk_len
    cuts = []
    for i in range(silent.shape[1]):
        if silent[0][i:i+min_cut_len].all():
            cuts.append(True)
        else:
            cuts.append(False)

    count = 0
    seg_indices = []
    for i in range(len(cuts)):
        if cuts[i]:
            if count > min_segment_len // chunk_len:
                seg_indices.append([i-count, i])
            count = 0
            continue
        count += 1
    # last segment
    if count > min_segment_len // chunk_len:
        seg_indices.append([len(cuts)-count, len(cuts)])

    out = []
    for (i, j) in seg_indices:
        out.append(wav[:, i*hop_len:j*hop_len].numpy())

    return out

def plot_waveform(waveform, sample_rate, stem="waveform"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(stem)
    figure.savefig(f"{stem}.png")

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    figure.savefig("specgram.png")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default='./data/artist20/mp3s-32k')
    parser.add_argument('--save_path', type=Path, default='./data/artist20/mp3s-32k-seperated')
    parser.add_argument('--model', type=str, default='htdemucs')
    args = parser.parse_args()
    return args

def preprocess():
    args = parse_args()
    song_exts = ['mp3', 'wav']
    song_files = []
    for ext in song_exts:
        song_files.extend(args.data_path.glob(f'**/*.{ext}'))
    song_files = sorted(song_files)

    # seperate vocals
    for song_file in song_files:
        seperate_vocals(
            song_file,
            args.data_path,
            args.save_path,
            args.model,
        )

    # generate singer list
    singers = set()
    for entry in (args.data_path.parent / 'train.tsv').read_text().splitlines():
        _, singer, _ = entry.split('\t')
        singers.add(singer)
    for entry in (args.data_path.parent / 'validation.tsv').read_text().splitlines():
        _, singer, _ = entry.split('\t')
        singers.add(singer)
    singers = sorted(list(singers))
    singer_file = args.data_path.parent.parent / 'singer.txt'
    singer_file.write_text('\n'.join(singers))

    # generate dataset cache
    _ = HW1Dataset(args.data_path.parent / 'train.tsv', singer_file)
    _ = HW1Dataset(args.data_path.parent / 'validation.tsv', singer_file)

def seperate_vocals(
    song_file: Path,
    data_path: Path,
    save_path: Path,
    model: str,
):
    tgt_path = save_path / song_file.relative_to(data_path)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_vocal_path = tgt_path.parent / (tgt_path.stem + '-vocals.wav')
    tgt_no_vocal_path = tgt_path.parent / (tgt_path.stem + '-no_vocals.wav')

    if tgt_vocal_path.exists() and tgt_no_vocal_path.exists():
        print(f"Allready exists: {tgt_vocal_path} and {tgt_no_vocal_path} ... Skipping")
        return

    sep_dir = Path(f"./separated/{model}")
    shutil.rmtree(sep_dir, ignore_errors=True)
    sep_dir.mkdir(parents=True, exist_ok=True)

    demucs.separate.main(["--two-stems", "vocals", "-n", "htdemucs", str(song_file)])

    vocal_file = list(Path("./separated").glob('**/vocals.wav'))
    no_vocal_file = list(Path("./separated").glob('**/no_vocals.wav'))
    assert len(vocal_file) == 1 and len(no_vocal_file) == 1
    vocal_file = vocal_file[0]
    no_vocal_file = no_vocal_file[0]

    vocal_file.replace(tgt_vocal_path)
    no_vocal_file.replace(tgt_no_vocal_path)

if __name__ == '__main__':
    preprocess()
