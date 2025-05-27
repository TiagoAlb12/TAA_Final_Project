import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=4, filters=[32, 64, 128], dropout_rates=[0.2, 0.3, 0.4]):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(dropout_rates[0])

        self.conv3 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(filters[1])
        self.conv4 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(dropout_rates[1])

        self.conv5 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(filters[2])
        self.conv6 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(dropout_rates[2])

        self.flatten_dim = filters[2] * 28 * 28  # Assuming input 224x224 and 3x MaxPool2d(2)
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x