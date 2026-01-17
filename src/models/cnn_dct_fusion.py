import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

class CNN_DCT_Fusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.densenet121(
            weights=DenseNet121_Weights.DEFAULT
        )
        self.cnn.classifier = nn.Identity()

        self.dct_fc = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, img, dct):
        cnn_feat = self.cnn(img)
        dct_feat = self.dct_fc(dct)
        fused = torch.cat([cnn_feat, dct_feat], dim=1)
        return self.classifier(fused)
