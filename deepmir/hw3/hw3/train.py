import time
import argparse
import json
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from tqdm import tqdm

from .utils import AttrDict, init_ckpt_dir, save_checkpoint, scan_checkpoint, load_checkpoint
from .repr import Tokenizer
from .data import AILabs1k7Dataset
from .model import RemiTransformer

# warnings.simplefilter(action='ignore', category=FutureWarning)

def train(ca, hp):
    torch.cuda.manual_seed(hp.seed)
    device = torch.device('cuda:{:d}'.format(ca.cuda_device_id))

    print("Loading dataset...")
    print("Vocab file : ", ca.vocab_file)
    print("Data directory : ", ca.data_dir)
    tokenizer = Tokenizer(ca.vocab_file)
    dataset = AILabs1k7Dataset(ca.data_dir, tokenizer)
    trainset = torch.utils.data.Subset(dataset, range(1, len(dataset)))
    validset = torch.utils.data.Subset(dataset, [0])

    print("Dataset loaded.")

    model = RemiTransformer(hp)
    print("Model loaded :")
    print(model)
    print("Model parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("checkpoints directory : ", ca.checkpoint_path)

    last_epoch = -1
    if ca.checkpoint_path.is_dir():
        ckpt = scan_checkpoint(ca.checkpoint_path, 'model_')
    else:
        ckpt = None

    steps = 0
    if ckpt is None:
        state_dict = None
        last_epoch = -1
    else:
        state_dict = load_checkpoint(ckpt, device)
        model.load_state_dict(state_dict['model'])
        steps = state_dict['steps'] + 1
        last_epoch = state_dict['epoch']

    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), hp.learning_rate, betas=[hp.adam_b1, hp.adam_b2])
    if state_dict is not None:
        optim.load_state_dict(state_dict['optim'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=hp.sched_T, eta_min=hp.learning_rate_min, last_epoch=last_epoch)

    train_loader = AILabs1k7Dataset.get_dataloader(
        tokenizer, hp.max_seq_len,
        trainset, num_workers=mp.cpu_count(), shuffle=True, batch_size=ca.batch_size
    )
    valid_loader = AILabs1k7Dataset.get_dataloader(
        tokenizer, hp.max_seq_len,
        validset, num_workers=mp.cpu_count(), batch_size=1,
    )

    sw = SummaryWriter(ca.checkpoint_path / 'logs')

    model.train()
    for epoch in range(max(0, last_epoch), ca.training_epochs):
        start = time.time()
        print("Epoch: {}".format(epoch+1))

        pbar = tqdm(desc=f'Epoch {epoch+1}', total=len(trainset), dynamic_ncols=True)
        for _, batch in enumerate(train_loader):
            song_ids = batch['song_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            out = model(
                input_ids = song_ids,
                labels = label_ids,
            )
            optim.zero_grad()
            loss = out.loss
            loss.backward()
            optim.step()

            if steps % ca.stdout_interval == 0:
                pbar.set_postfix_str(f'step={steps}, loss={loss.item():4.3f}')

            # checkpointing
            if steps % ca.checkpoint_interval == 0 and steps != 0:
                # checkpoint_path = "{}/model_{:08d}".format(ca.checkpoint_path, steps)
                checkpoint_path = ca.checkpoint_path / f'model_{steps:08d}'
                save_checkpoint(checkpoint_path, 
                                {'model': model.state_dict(),
                                 'optim': optim.state_dict(),
                                 'steps': steps,
                                 'epoch': epoch})

            # Tensorboard summary logging
            if steps % ca.summary_interval == 0:
                sw.add_scalar("training/loss", loss, steps)

            # Validation
            if steps % ca.validation_interval == 0:  # and steps != 0:
                model.eval()
                torch.cuda.empty_cache()
                model.train()

            steps += 1
            pbar.update(song_ids.shape[0])

            if steps > ca.training_steps:
                break
        pbar.close()

        if steps > ca.training_steps:
            break

        scheduler.step()
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=Path, required=True)
    parser.add_argument('--config_file', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--vocab_file', type=Path, required=True)
    parser.add_argument('--training_epochs', default=1000, type=int)
    parser.add_argument('--training_steps', default=300000, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    parser.add_argument('--stdout_interval', default=1, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--cuda_device_id', default=0, type=int)
    ca = parser.parse_args()

    json_config = json.loads(ca.config_file.read_text())
    hp = AttrDict(json_config)
    init_ckpt_dir(ca.checkpoint_path, ca.config_file)

    torch.manual_seed(hp.seed)
    train(ca, hp)

if __name__ == '__main__':
    main()
