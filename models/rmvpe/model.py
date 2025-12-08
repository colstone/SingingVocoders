from torch import nn
from .deepunet import DeepUnet

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class BiLSTM(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]
    

class E2E(nn.Module):
    def __init__(self, config, kernel_size = (2,2)):
        super(E2E, self).__init__()
        n_blocks = config['n_blocks']
        en_de_layers = config['en_de_layers']
        inter_layers = config['inter_layers']
        in_channels = config['in_channels']
        en_out_channels = config['en_out_channels']
        n_gru = config['n_gru']
        n_mels = config['audio_num_mel_bins']
        n_class = config['n_class']
        self.unet = DeepUnet(n_mels, kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * n_mels, 256, n_gru),
                nn.Linear(512, n_class),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * n_mels, n_class),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        # x = self.fc(x)
        hidden_vec = 0
        if len(self.fc) == 4:
            for i in range(len(self.fc)):
                x = self.fc[i](x)
                if i == 0:
                    hidden_vec = x
        return hidden_vec, x

