import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch


class FramesClassifier(BaseModel):
    def __init__(self, num_classess=7) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3,3), stride=(3,3))
        self.bn1 = nn.BatchNorm2d(10)
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2, 2))
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=25, kernel_size=(3,3), stride=(2,2))
        self.bn2 = nn.BatchNorm2d(25)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv3 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(5,5), stride=(2,2))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.drop2 = nn.Dropout(0.5)
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=60, kernel_size=(3,3), stride=(3,3))
        self.bn4 = nn.BatchNorm2d(60)
        self.fc1 = nn.Linear(480, 250)
        self.drop4 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(250, 50)
        self.drop5 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(50, num_classess)
        
    def forward(self, x):
        x=self.bn1(F.relu(self.conv1(x)))
        x=self.drop1(x)
        x=self.bn2(F.relu(self.conv2(x)))
        x=self.pool1(x)


        x=self.bn3(F.relu(self.conv3(x)))
        x=self.drop2(x)
        x=self.bn4(F.relu(self.conv4(x)))
        x=self.pool2(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        x = F.relu(self.fc3(x))
        x = self.drop5(x)
        x = self.fc4(x)

        return x