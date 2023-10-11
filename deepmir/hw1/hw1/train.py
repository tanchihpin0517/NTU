import argparse
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import multiprocessing as mp

from .data import HW1Dataset
from .model import CNNResModel

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=Path, default='./data/artist20')
    args.add_argument('--singer_file', type=Path, default='./data/singer.txt')
    args.add_argument('--batch_size', type=int, default=4, help='batch size')
    args.add_argument('--ckpt_dir', type=Path, required=True, help='checkpoint directory')
    args.add_argument('--ckpt_file', type=str, help='checkpoint file')
    args.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
    args.add_argument('--num_gpu', default=1, help='number of gpus')
    args.add_argument('--debug', action='store_true', help='debug mode')

    return args.parse_args()

def main():
    args = parse_args()

    train_dataset = HW1Dataset(
        args.data_path / 'train.tsv',
        args.singer_file,
    )
    valid_dataset = HW1Dataset(
        args.data_path / 'validation.tsv',
        args.singer_file,
    )

    train_loader = train_dataset.get_dataloader(
        args.batch_size,
        shuffle=True if not args.debug else False,
        num_workers=mp.cpu_count() if not args.debug else 1,
    )
    valid_loader = valid_dataset.get_dataloader(
        args.batch_size,
        shuffle=False,
        num_workers=mp.cpu_count() if not args.debug else 1,
    )

    model_config = CNNResModel.default_config()
    model_config['n_class'] = len(train_dataset.singers)
    model = CNNResModel(model_config)

    checkpoint_callback = ModelCheckpoint(
        # filename='{epoch:02d}-{tl_rec:.2f}-{tl_cls:.2f}-{vl_rec:.2f}-{vl_cls:.2f}-{step}',
        filename='{epoch:03d}-{tl:.2f}-{v1:.2f}-{v3:.2f}-{step}',
        save_top_k=-1,
    )
    trainer = L.Trainer(
        devices=args.num_gpu,
        accelerator="gpu",
        precision="16-mixed",
        max_epochs=args.num_epochs,
        default_root_dir=Path(args.ckpt_dir),
        callbacks=[checkpoint_callback],
        # logger=logger,
        fast_dev_run=10 if args.debug else False,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.ckpt_file,
    )



if __name__ == "__main__":
    main()
