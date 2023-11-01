import torchaudio
from pathlib import Path
import shutil

def resample(in_dir, out_dir, tgt_sr=22050):
    for in_file in in_dir.glob("**/*"):
        if in_file.is_dir():
            continue

        out_file = out_dir / in_file.relative_to(in_dir)

        if out_file.exists():
            continue

        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.suffix == ".wav":
            print("resample:", in_file, "->", out_file)
            audio, sr = torchaudio.load(in_file)
            if sr != tgt_sr:
                audio = torchaudio.transforms.Resample(sr, tgt_sr)(audio)
            torchaudio.save(out_file, audio, tgt_sr)
        else:
            print("copy:", in_file, "->", out_file)
            shutil.copy(in_file, out_file)
            

if __name__ == '__main__':
    in_dir = Path("./data/m4singer")
    out_dir = Path("./data/m4singer_22050")
    resample(in_dir, out_dir)

    in_dir = Path("./data/m4singer_valid")
    out_dir = Path("./data/m4singer_valid_22050")
    resample(in_dir, out_dir)
