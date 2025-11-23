import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels, stride=1, channels_upsample=False):
        super(ResBlock, self).__init__()
        self.identity_shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if channels_upsample else nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h_x = self.conv1(x)
        h_x = self.bn1(h_x)
        h_x = self.relu(h_x)
        h_x = self.conv2(h_x)
        h_x = self.bn2(h_x)
        h_x = self.relu(h_x)
        h_x = self.conv3(h_x)
        h_x = self.bn3(h_x)
        x = self.identity_shortcut(x)
        return self.relu(h_x + x)
    
class ResNet(nn.Module):
    def __init__(self, conv_layers, num_classes):
        super(ResNet, self).__init__()
        self.conv_layers = conv_layers
        assert len(conv_layers) == 4, "ResNet need to downsample exactly 4 times"

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(idx_conv_layer=0, first_layer_stride=1),
        )
        self.conv3_x = self._make_layer(idx_conv_layer=1, first_layer_stride=2)
        self.conv4_x = self._make_layer(idx_conv_layer=2, first_layer_stride=2)
        self.conv5_x = self._make_layer(idx_conv_layer=3, first_layer_stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)


    def forward(self, x):
        # print(f"x.shape : {x.shape}")
        x = self.conv1(x)
        # print(f"x.shape : {x.shape}")
        x = self.conv2_x(x)
        # print(f"x.shape : {x.shape}")
        x = self.conv3_x(x)
        # print(f"x.shape : {x.shape}")
        x = self.conv4_x(x)
        # print(f"x.shape : {x.shape}")
        x = self.conv5_x(x)
        # print(f"x.shape : {x.shape}")
        x = self.avg_pool(x)
        # print(f"x.shape : {x.shape}")
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


    def _make_layer(self, idx_conv_layer, first_layer_stride, channels_upsample=True):
        max_resblock_number = self.conv_layers[idx_conv_layer]
        layers = []

        intermediate_channels = 64 * 2**idx_conv_layer
        out_channels = intermediate_channels*4
        in_channels = 64 if idx_conv_layer==0 else intermediate_channels*2
 
        for resblock_number in range(max_resblock_number):
            if resblock_number == 0 :
                layers.append(
                    ResBlock(in_channels=in_channels,
                             intermediate_channels=intermediate_channels,
                             out_channels=out_channels,
                             stride=first_layer_stride,
                             channels_upsample=channels_upsample))
            else:
                layers.append(
                    ResBlock(in_channels=out_channels,
                             intermediate_channels=intermediate_channels,
                             out_channels=out_channels,
                             stride=1,
                             channels_upsample=False))
        
        return nn.Sequential(*layers)
    
def ResNet50(num_classes):
    return ResNet(conv_layers=[3, 4, 6, 3], num_classes=num_classes)

def ResNet101(num_classes):
    return ResNet(conv_layers=[3, 8, 36, 3], num_classes=num_classes)

def test():
    model = ResNet50(num_classes=100)
    model = model.to('cuda')
    print(model)
    x = torch.randn((2**8,3,300,300), device='cuda')
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()