import torch
import torch.nn as nn

class PTRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.input_shape = (None, 5)
        self.output_shape = (None, 1)
    
    def forward(self, x):
        return self.layers(x)

class PTImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 8 * 8, 10)
        self.input_shape = (None, 3, 32, 32)
        self.output_shape = (None, 10)
        self.num_classes = 10
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 8 * 8)
        return self.fc(x) 