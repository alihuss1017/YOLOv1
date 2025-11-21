from pretrain.model import ClassifierConvNet
import torch.nn as nn

class DetectionConvNet(nn.Module):

    '''
    Detection-tasked convNet. Modifies ClassifierConvNet's fcn layer 
    and applies random weight initialization to conv and linear layers inside the fcn. 

    Inputs:
        convNet: ClassifierConvNet
    '''

    def __init__(self, convNet: ClassifierConvNet):
        super().__init__()
        self.convNet = convNet
        # ---------- modifying fcn to perform detection task ---------- #
        self.convNet.fc = nn.Sequential(
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
            nn.Linear(4096, 7 * 7 * 210),
            nn.Sigmoid(),
        )
        # ----------------------------------------------------------- #

        self.convNet.fc.apply(self._init_weights)

    # ---------- applying random weight initialization to layers in fcn  ---------- #
    def _init_weights(self, layer):
        ''' applies weights only to linear or conv2D layers '''
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    # ----------------------------------------------------------------------------- #

    def forward(self, x):
        out = self.convNet(x)
        return out.reshape(64, 7, 7, 210)
    