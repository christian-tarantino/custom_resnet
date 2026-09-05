import torchvision.models as models
import torch.nn as nn

def get_cifar_resnet18(num_classes=10):
    # Prende la ResNet-18 ufficiale PyTorch
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    model.maxpool = nn.Identity()# type: ignore
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model