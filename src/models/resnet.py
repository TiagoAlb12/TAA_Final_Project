import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet18(num_classes=4, grayscale=True, freeze_until=6):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    # Adapt the first convolutional layer for grayscale images
    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze layers until the specified index
    for i, (name, param) in enumerate(model.named_parameters()):
        param.requires_grad = (i >= freeze_until)

    return model