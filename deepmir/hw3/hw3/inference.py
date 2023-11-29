import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import time

from .utils import AttrDict, load_checkpoint
from . import utils
from .repr import Tokenizer
from .model import RemiTransformer


def inference(ca, hp):
    device = torch.device('cuda:{:d}'.format(ca.cuda_device_id))
    tokenizer = Tokenizer(ca.vocab_file)

    model = RemiTransformer(hp)
    state_dict = load_checkpoint(ca.checkpoint_file, device)
    model.load_state_dict(state_dict['model'])

    model.eval()
    model.to(device)
    sampling_method = utils.top_p if ca.sampling_method == 'top-p' else utils.top_k
    pbar = tqdm(total=ca.num_gen_song, dynamic_ncols=True)
    for song_i in range(ca.num_gen_song):
        ca.output_dir.mkdir(exist_ok=True)
        # dir_name = f'{ca.checkpoint_file.stem.split("_")[-1]}_{ca.sampling_method}_{ca.temperature}_{ca.threshold}' + ('_sink' if ca.sink_attention else '_nosink')
        dir_name = f'{ca.checkpoint_file.stem.split("_")[-1]}_{ca.sampling_method}_{ca.temperature}_{ca.threshold}_{ca.strategy}_{hp.max_seq_len}'
        output_dir = ca.output_dir / dir_name
        output_dir.mkdir(exist_ok=True)
        out_midi_file = output_dir / f'{song_i:03d}_midi.mid'
        out_event_file = output_dir / f'{song_i:03d}_events.txt'
        out_stats_file = output_dir / f'{song_i:03d}_stats.txt'

        if out_midi_file.exists() and out_event_file.exists() and out_stats_file.exists():
            pbar.update(1)
            continue

        start_time = time.time()
        out_events = ['spec_ss']
        out_ids = [tokenizer.e2i('spec_ss')]
        input_ids = torch.LongTensor([tokenizer.e2i('spec_bos')] + out_ids).unsqueeze(0).to(device)
        kv_cache = None
        bar_count = 0

        pbar.set_description_str(f"{(0, 0)}")
        for _ in range(10240):
            out = model(
                input_ids = input_ids,
                past_key_values = kv_cache,
            )
            logits = out.logits[:, -1, :]
            logits = sampling_method(logits, ca.threshold)
            out_id = torch.multinomial(F.softmax(logits / ca.temperature, dim=-1), num_samples=1)
            out_event = tokenizer.i2e(out_id.item())

            out_events.append(out_event)
            out_ids.append(out_id.item())

            if out_event == 'bar':
                bar_count += 1
                if bar_count > ca.num_gen_bar:
                    break
            if out_event == 'spec_se':
                if bar_count == ca.num_gen_bar:
                    break
                else:
                    # reset
                    start_time = time.time()
                    out_events = ['spec_ss']
                    out_ids = [tokenizer.e2i('spec_ss')]
                    input_ids = torch.LongTensor(
                        [tokenizer.e2i('spec_bos')] + out_ids
                    ).unsqueeze(0).to(device)
                    kv_cache = None
                    bar_count = 0

            if ca.strategy == 'nocache':
                input_ids = torch.cat([input_ids, out_id], dim=-1)
                if input_ids.shape[1] > hp.max_seq_len - 1: # n-1
                    # bos = torch.LongTensor(tokenizer.e2i('spec_bos')).unsqueeze(0).to(device)
                    # input_ids = torch.cat([bos, input_ids[:, -hp.max_seq_len+2:]], dim=1)
                    input_ids = torch.LongTensor(
                        [tokenizer.e2i('spec_bos')] + out_ids[-hp.max_seq_len+2:]
                    ).unsqueeze(0).to(device)
                    # assert input_ids.shape[1] == hp.max_seq_len
            elif ca.strategy == 'stride':
                input_ids = out_id
                kv_cache = out.past_key_values
                kv_len = kv_cache[0][0].shape[2]
                if kv_len > (hp.max_seq_len - 1) - 1: # n-1
                    shift = (hp.max_seq_len - hp.max_seq_len // 8)
                    input_ids = torch.LongTensor(
                        [tokenizer.e2i('spec_bos')] + out_ids[-shift+1:]
                    ).unsqueeze(0).to(device)
                    kv_cache = None
            elif ca.strategy == 'nobos':
                input_ids = out_id
                kv_cache = out.past_key_values
                kv_len = kv_cache[0][0].shape[2]
                if kv_len > (hp.max_seq_len - 1) - 1: # n-1
                    shift = (hp.max_seq_len - hp.max_seq_len // 8)
                    input_ids = torch.LongTensor(out_ids[-shift:]).unsqueeze(0).to(device)
                    kv_cache = None
            elif ca.strategy == 'window':
                input_ids = out_id
                kv_cache = []
                for layer in out.past_key_values:
                    kv = []
                    for cache in layer:
                        kv.append(cache[:, :, -hp.max_seq_len+2:, :])
                    kv_cache.append(tuple(kv))
                kv_cache = tuple(kv_cache)
            elif ca.strategy == 'sink':
                input_ids = out_id
                kv_cache = out.past_key_values
                kv_len = kv_cache[0][0].shape[2]
                if kv_len > (hp.max_seq_len - 1) - 1: # n-1
                    new_kv_cache = []
                    for layer in out.past_key_values:
                        kv = []
                        for cache in layer:
                            sink_size = 16
                            cache = torch.cat([
                                cache[:, :, :sink_size, :],
                                cache[:, :, -hp.max_seq_len+2+sink_size:, :]
                            ], dim=2)
                            kv.append(cache)
                        new_kv_cache.append(tuple(kv))
                    kv_cache = tuple(new_kv_cache)
            else:
                raise ValueError(f"Unknown strategy: {ca.strategy}")

            pbar.set_description_str(f"{(bar_count, len(out_events))}")

        midi = tokenizer.events_to_midi(out_events, 4)
        midi.dump(out_midi_file)
        out_event_file.write_text('\n'.join(out_events))
        elapsed_time = time.time() - start_time
        stats = {
            'elapsed_time': elapsed_time,
            'num_events': len(out_events),
            'num_bars': bar_count,
            'events_per_sec': len(out_events) / elapsed_time,
            'bars_per_sec': bar_count / elapsed_time,
        }
        out_stats_file.write_text(json.dumps(stats, indent=2))
        pbar.update(1)
    pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=Path, required=True)
    parser.add_argument('--checkpoint_file', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--vocab_file', type=Path, required=True)
    parser.add_argument('--num_gen_bar', default=32, type=int)
    parser.add_argument('--num_gen_song', default=20, type=int)
    parser.add_argument('--max_seq_len', type=int)
    parser.add_argument('--sampling_method', type=str, required=True, choices=['top-p', 'top-k'])
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--threshold', default=0.9, type=float)
    parser.add_argument('--cuda_device_id', default=0, type=int)
    # parser.add_argument('--sink_attention', action='store_true')
    # parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--strategy', type=str, required=True, choices=['nocache', 'stride', 'nobos', 'window', 'sink'])
    ca = parser.parse_args()

    json_config = json.loads(ca.config_file.read_text())
    hp = AttrDict(json_config)
    if ca.max_seq_len is not None:
        hp.max_seq_len = ca.max_seq_len

    # torch.manual_seed(hp.seed)
    with torch.no_grad():
        inference(ca, hp)

if __name__ == '__main__':
    main()
