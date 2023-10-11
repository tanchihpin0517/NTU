from pathlib import Path
import shutil
import demucs.separate

def main(
    data_path: Path = Path('./data/artist20_testing_data'),
    sep_path: Path = Path('./data/artist20_testing_data-seperated'),
):
    files = list(data_path.glob('*'))
    for file in files:
        seperate_vocals(file, sep_path, 'htdemucs')

def seperate_vocals(
    song_file: Path,
    tgt_path: Path,
    model: str,
):
    tgt_path.mkdir(parents=True, exist_ok=True)
    tgt_vocal_path = tgt_path / (song_file.stem + '-vocals.wav')
    tgt_no_vocal_path = tgt_path / (song_file.stem + '-no_vocals.wav')

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
    main()
