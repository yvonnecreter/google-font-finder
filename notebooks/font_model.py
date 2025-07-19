import torch.nn as nn
from transformers import VisionEncoderDecoderModel

EMBED_D=128

class FontDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
        self.convolutions = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=3//2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=128,
                kernel_size=9,
                stride=1,
                padding=9//2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2)
        )
        self.projection = nn.Sequential(
            nn.Linear(128, 128), 
            nn.GELU(), 
            nn.Linear(128, EMBED_D))

    def forward(self, x):
        x = self.encoder(x)
        x = self.convolutions(x)
        x = self.projection(x)
        return x.view(x.shape[0], -1) # -1 := inferred from other dimensions # BxD