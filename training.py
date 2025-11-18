from model import ConvNet
import torch.nn as nn
import torch

convNet = ConvNet()

convNet.fc = nn.Sequential(
    nn.Conv2d(1024, 1024, (3,3), padding = 1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(1024, 1024, (3,3), padding = 1, stride = 2),
    nn.LeakyReLU(0.1),
    nn.Conv2d(1024, 1024, (3,3), padding = 1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(1024, 1024, (3,3), padding = 1),
    nn.LeakyReLU(0.1),
    nn.Flatten(start_dim = 1, end_dim = -1),
    nn.Linear(7 * 7 * 1024, 4096),
    nn.LeakyReLU(0.1),
    nn.Linear(4096, 7 * 7 * 210)
)

def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
convNet.fc.apply(init_weights)
test_output = convNet(torch.randn(1, 3, 448, 448))
test_output = test_output.reshape(1, 7, 7, 210)
print(test_output.shape)