import torch.nn as nn

"""
Input output shapes
text (BERT): (50, 768)
text: (50, 300s)
label: (1) -> [sentiment]
"""


class BERTTextLSTMNet(nn.Module):
    def __init__(self, features_only=False):
        super(BERTTextLSTMNet, self).__init__()
        self.features_only = features_only
        self.lstm1 = nn.LSTM(
            input_size=768, hidden_size=128, num_layers=1, batch_first=True
        )
        # self.lstm1 = nn.Linear(768,128)
        self.lstm2 = nn.LSTM(
            input_size=128, hidden_size=32, num_layers=1, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x, _ = self.lstm1(x)
        last = x[:, -1, :]

        return self.classifier(last)


