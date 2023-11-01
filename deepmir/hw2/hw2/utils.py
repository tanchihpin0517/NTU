import numpy as np
import torch
import torchaudio
import librosa
import os
import pickle
from scipy.io.wavfile import write

N_FFT=1024
NUM_MELS=80
SAMPLING_RATE=22050
HOP_SIZE=256
WIN_SIZE=1024
FMIN=0
FMAX=8000

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft=N_FFT, num_mels=NUM_MELS, sampling_rate=SAMPLING_RATE, hop_size=HOP_SIZE, win_size=WIN_SIZE, fmin=FMIN, fmax=FMAX, center=False, norm=True):
    device = y.device
    melTorch = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_fft=n_fft, n_mels=num_mels, \
           hop_length=hop_size, win_length=win_size, f_min=fmin, f_max=fmax, pad=int((n_fft-hop_size)/2), center=center).to(device)      
    spec = melTorch(y)
    if norm:
        spec = spectral_normalize_torch(spec)
    return spec

def griffin_lim(wav):
    wav = wav.numpy()
    spec = librosa.stft(wav, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=WIN_SIZE)
    spec = np.abs(spec)
    out = librosa.griffinlim(spec, hop_length=HOP_SIZE, win_length=WIN_SIZE, n_fft=N_FFT)
    out = torch.from_numpy(out)
    return out

def to_mono(audio, dim=-2): 
    if len(audio.size()) > 1:
        return torch.mean(audio, dim=dim, keepdim=True)
    else:
        return audio

def load_audio(audio_path, sr=SAMPLING_RATE, mono=True):
    if 'mp3' in audio_path:
        torchaudio.set_audio_backend('sox_io')
    audio, org_sr = torchaudio.load(audio_path)

    audio = to_mono(audio) if mono else audio

    if sr and org_sr != sr:
        audio = torchaudio.transforms.Resample(org_sr, sr)(audio)

    return audio, sr if sr else org_sr

def save_audio(audio_path, audio, sr=SAMPLING_RATE):
    torchaudio.save(audio_path, audio, sr)

def pickle_dump(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# if __name__ == '__main__':
#     load_audio_path = '/path/to/your/audio/dir'
#     save_npy_path = '/path/you/want/to/save/mel_npy'
#     if not os.path.exists(save_npy_path):
#         os.mkdir(save_npy_path)
#     audio_list = os.listdir(load_audio_path)
#     audio_list.sort()
#     for audio in audio_list:
#         y = load_audio(os.path.join(load_audio_path, audio), sr=sampling_rate)
#         mel_tensor = mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
#         mel = mel_tensor.squeeze().cpu().numpy()
#         file_name = os.path.join(save_npy_path, audio[:-4]+'.npy')
#         np.save(file_name, mel)
#         mel = np.load(file_name) # check the .npy is readable
#
#     # plot the last melspectrogram
#     # ref: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     # don't forget to do dB conversion
#     S_dB = librosa.power_to_db(S, ref=np.max)
#     img = librosa.display.specshow(S_dB, x_axis='time',
#                              y_axis='mel', sr=sampling_rate,
#                              fmax=fmax, ax=ax, hop_length=hop_size, n_fft=n_fft)
#     fig.colorbar(img, ax=ax, format='%+2.0f dB')
#     ax.set(title='Mel-frequency spectrogram')
