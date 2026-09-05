from torch import nn
import torch









class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Conv di adattamento per il shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x):
        res = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + res 
        return self.act2(out)
    


class ResNet_tiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.prep = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU()
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16, stride=1),
            ResidualBlock(16, 16, stride=1),
            ResidualBlock(16, 16, stride=1)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1)
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(128, num_classes)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)       

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)