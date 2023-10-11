import torch
from torch import nn
import lightning as L
import torchaudio
import torch.nn.functional as F

class Res_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super(Res_2d, self).__init__()
        # convolution
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out

class CNNResModel(L.LightningModule):
    @classmethod
    def default_config(cls):
        return {
            'n_channels': 128,
            'sample_rate': 16000,
            'n_fft': 512,
            'f_min': 0.0,
            'f_max': 8000.0,
            'n_mels': 128,
            'top_db': 80.0,
        }

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        n_channels = config['n_channels']
        n_class = config['n_class']
        sample_rate = config['sample_rate']
        n_fft = config['n_fft']
        f_min = config['f_min']
        f_max = config['f_max']
        n_mels = config['n_mels']
        top_db = config['top_db']

        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)
        self.spec_bn = nn.BatchNorm2d(1)
        self.config = config

        # CNN
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer5 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer6 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer7 = Res_2d(n_channels*2, n_channels*4, stride=2)

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x

    def training_step(self, batch, _):
        y = self(batch['frames'])
        loss = F.cross_entropy(y, batch['labels'])
        self.log('tl', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        y = self(batch['frames'])
        dist = F.softmax(y, dim=-1)
        top1 = torch.argmax(dist, dim=-1)
        v1 = torch.sum(top1 == batch['labels']) / len(batch['labels'])
        self.log('v1', v1, prog_bar=True)

        top3 = torch.topk(dist, 3, dim=-1).indices
        v3 = torch.sum(torch.any(top3 == batch['labels'].unsqueeze(-1), dim=-1)) / len(batch['labels'])
        self.log('v3', v3, prog_bar=True)
        # return v1

        # loss = F.cross_entropy(y, batch['labels'])
        # self.log('vl', loss, prog_bar=True)
        # return 3

    def predict(self, song_frames):
        assert song_frames.dim() == 3, song_frames.shape # (n_frames, channel, wav)
        dist = F.softmax(self(song_frames), dim=-1)
        accum = dist.sum(dim = 0)
        top3 = torch.topk(accum, 3, dim=-1).indices
        return top3.tolist()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


