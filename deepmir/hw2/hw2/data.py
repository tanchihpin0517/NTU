import torch
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import random

from . import utils

class VocalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, segment_size):
        self.data_dir = data_dir
        self.data = []

        cache_path = data_dir.parent / f"{data_dir.stem}_cache.pkl"
        if cache_path.exists():
            print("Loading cached data...")
            self.data = utils.pickle_load(cache_path)
            return

        exts = ["*.wav", "*.mp3"]
        self.audio_paths = []
        for ext in exts:
            self.audio_paths.extend(list(data_dir.glob(f"**/{ext}")))

        # with mp.Pool(1) as pool:
        #     print("Loading audio files...")
        #     pbar = tqdm(total=len(self.audio_paths))
        #     for result in pool.imap(self.proc_audio, self.audio_paths):
        #         self.data.append(result)
        #         pbar.update()
        #     pbar.close()
        for audio_path in tqdm(self.audio_paths):
            self.data.append(self.proc_audio(audio_path))

        utils.pickle_dump(self.data, cache_path)

    @staticmethod
    def proc_audio(audio_path):
        audio, norm_factor = utils.load_audio(str(audio_path))
        return {
            "audio": audio,
            "norm_factor": norm_factor,
            "audio_path": audio_path
        }

    def __getitem__(self, index):
        data = self.data[index]
        audio = data["audio"]
        audio_path = data["audio_path"]
        mel_loss = utils.mel_spectrogram(audio)

        if audio.size(1) >= self.segment_size:
            max_audio_start = audio.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start+self.segment_size]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel = utils.mel_spectrogram(audio)
        mel_loss = utils.mel_spectrogram(audio, fmax=None)

        return mel.squeeze(0), audio.squeeze(0), audio_path, mel_loss.squeeze(0)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    data_dirs = [
        Path("./data/m4singer"),
        Path("./data/m4singer_valid"),
    ]
    for data_dir in data_dirs:
        dataset = VocalDataset(data_dir)
