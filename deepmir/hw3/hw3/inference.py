import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

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
        dir_name = f'{ca.checkpoint_file.stem.split("_")[-1]}_{ca.sampling_method}_{ca.temperature}_{ca.threshold}' + ('_sink' if ca.sink_attention else '_nosink')
        output_dir = ca.output_dir / dir_name
        output_dir.mkdir(exist_ok=True)
        out_midi_file = output_dir / f'{song_i:03d}_midi.mid'
        out_event_file = output_dir / f'{song_i:03d}_events.txt'

        if out_midi_file.exists() and out_event_file.exists():
            pbar.update(1)
            continue

        out_events = ['spec_ss']
        out_ids = [tokenizer.e2i('spec_ss')]
        input_ids = torch.LongTensor([tokenizer.e2i('spec_bos')] + out_ids).unsqueeze(0).to(device)
        kv_cache = None
        bar_count = 0

        pbar.set_description_str(f"{(0, 0)}")
        for _ in range(102400):
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
                    out_events = ['spec_ss']
                    out_ids = [tokenizer.e2i('spec_ss')]
                    input_ids = torch.LongTensor([tokenizer.e2i('spec_bos')] + out_ids).unsqueeze(0).to(device)
                    kv_cache = None
                    bar_count = 0

            if ca.use_cache:
                input_ids = out_id
                kv_cache = out.past_key_values
                kv_len = kv_cache[0][0].shape[2]
                if kv_len > hp.max_seq_len - 1:
                    shift = hp.max_seq_len // 8
                    if ca.sink_attention:
                        input_ids = torch.LongTensor([tokenizer.e2i('spec_bos')] + out_ids[-shift+1:]).unsqueeze(0).to(device)
                    else:
                        input_ids = torch.LongTensor(out_ids[-shift:]).unsqueeze(0).to(device)
                    kv_cache = None
            else:
                input_ids = torch.cat([input_ids, out_id], dim=-1)
                if input_ids.shape[1] > hp.max_seq_len:
                    shift = hp.max_seq_len // 8
                    if ca.sink_attention:
                        input_ids = torch.cat([tokenizer.e2i('spec_bos'), input_ids[:, -shift+1:]], dim=1)
                    else:
                        input_ids = input_ids[:, -hp.max_seq_len:]
            pbar.set_description_str(f"{(bar_count, len(out_events))}")

        midi = tokenizer.events_to_midi(out_events, 4)
        midi.dump(out_midi_file)
        out_event_file.write_text('\n'.join(out_events))
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
    parser.add_argument('--sink_attention', action='store_true')
    parser.add_argument('--use_cache', action='store_true')
    ca = parser.parse_args()

    json_config = json.loads(ca.config_file.read_text())
    hp = AttrDict(json_config)
    if ca.max_seq_len is not None:
        hp.max_seq_len = min(ca.max_seq_len, hp.max_seq_len)

    # torch.manual_seed(hp.seed)
    with torch.no_grad():
        inference(ca, hp)

if __name__ == '__main__':
    main()
