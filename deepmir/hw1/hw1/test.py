import argparse
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import multiprocessing as mp
import torch
from tqdm import tqdm

from .data import HW1Dataset, HW1TestingDataset
from .model import CNNResModel

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', type=Path, default='./data/artist20_testing_data')
    args.add_argument('--singer_file', type=Path, default='./data/singer.txt')
    args.add_argument('--result_file', type=Path, default='./d12942015.csv')
    args.add_argument('--ckpt_file', type=str, help='checkpoint file')
    args.add_argument('--debug', action='store_true', help='debug mode')

    return args.parse_args()

def main():
    args = parse_args()

    singer_list = args.singer_file.read_text().splitlines()

    test_dataset = HW1TestingDataset(
        args.data_dir,
    )
    test_loader = test_dataset.get_dataloader(
        1,
        num_workers=mp.cpu_count() if not args.debug else 1,
    )

    model = CNNResModel.load_from_checkpoint(args.ckpt_file)
    model.cuda()
    model.eval()

    out = []

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_dataset)):
            for song in batch:
                result = model.predict(song['frames'].to(model.device))
                pred_singers = [singer_list[i] for i in result]
                song_id = int(song['file'].stem)

                out.append(str(song_id) + ',' + ','.join(pred_singers))

    args.result_file.write_text('\n'.join(out))

if __name__ == "__main__":
    main()
