from model import ConvNet
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops.boxes import box_iou
from loss import YOLOLoss

# ---------- loading pretrained ConvNet classifier ---------- #
convNet = ConvNet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load('pretrained_model.pt', map_location = device)
convNet.load_state_dict(state_dict)
# ----------------------------------------------------------- #


# ---------- modifying fcn to perform detection task ---------- #
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
# ----------------------------------------------------------- #


# ---------- applying random weight initialization to layers in fcn  ---------- #

def init_weights(layer):
    ''' applies weights only to linear or conv2D layers '''
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
convNet.fc.apply(init_weights)

# ----------------------------------------------------------------------------- #

class Trainer:
    def __init__(self, model: ConvNet, train_loader: DataLoader, num_epochs: int,
                 lr: int, device: str):
        
        self.device = device
        self.model = model.to(device)
        self.train_loader = DataLoader
        self.num_epochs = num_epochs
        self.lr = lr
        self.loss_fn = YOLOLoss()

    def run(self):
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for images, labels in self.train_loader:
                optimizer.zero_grad()

                images, labels = images.to(self.device), labels.to(self.device)
                predictions = self.model(images)

                loss = self.loss_fn(predictions, labels)
                loss.backward()

                optimizer.step()

            print(f'Epoch: {epoch + 1}: Loss: {epoch_loss / len(self.train_loader)}')

    def save_model(self):
        '''saves pretrained classifier to pretrained_model.pt'''
        torch.save(self.model.state_dict(), 'model.pt')
        print('Saved detection model to model.pt')