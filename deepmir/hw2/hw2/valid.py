import argparse
from pathlib import Path
import torchaudio
from tqdm import tqdm
import numpy as np
import json

from .utils import mel_spectrogram, griffin_lim, load_audio, save_audio
from .eval.evaluate import evaluate

def prepare(input_dir, output_dir, gl_dir):
    wav_files = sorted(input_dir.glob('**/*.wav'))
    for file in tqdm(wav_files, desc="prepare mel"):
        mel_file = output_dir / file.relative_to(input_dir).with_suffix('.npy')
        if mel_file.exists():
            continue
        mel_file.parent.mkdir(parents=True, exist_ok=True)
        audio, sr = torchaudio.load(file)
        mel = mel_spectrogram(audio, norm=False)
        with open(mel_file, 'wb') as f:
            np.save(f, mel.numpy())

    for file in tqdm(wav_files, desc="griffin-lim"):
        gl_file = gl_dir / file.relative_to(input_dir).with_suffix('.wav')
        if gl_file.exists():
            continue
        gl_file.parent.mkdir(parents=True, exist_ok=True)
        audio, sr = load_audio(str(file))
        gl = griffin_lim(audio)
        save_audio(gl_file, gl, sr)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prepare', action='store_true')
    argparser.add_argument('--input_dir', type=Path)
    argparser.add_argument('--output_dir', type=Path)
    argparser.add_argument('--gl_dir', type=Path)
    argparser.add_argument('--gt_dir', type=Path)
    argparser.add_argument('--result_dir', type=Path)
    argparser.add_argument('--score_file', type=Path)
    args = argparser.parse_args()

    if args.prepare:
        prepare(args.input_dir, args.output_dir, args.gl_dir)
    else:
        results = evaluate(args.gt_dir, args.result_dir)
        with open(args.score_file, 'w') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)
