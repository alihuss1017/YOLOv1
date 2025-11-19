import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from train.model import DetectionConvNet
from utils.loss import YOLOLoss

class Trainer:

    '''
    Trainer instance for detection task. 

    Inputs:
        model: DetectionConvNet
        device: str
    '''
    def __init__(self, model: DetectionConvNet, device: str):
        self.device = device
        self.model = model.to(device)
        self.loss_fn = YOLOLoss()

    def run(self, num_epochs: int, train_loader: DataLoader):

        '''
        Runs the training algorithm, prints per epoch loss.
        
        Inputs:
            num_epochs: int
            train_loader: DataLoader
        '''
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            for images, labels in train_loader:
                optimizer.zero_grad()

                images, labels = images.to(self.device), labels.to(self.device)
                predictions = self.model(images)

                loss = self.loss_fn(predictions, labels)
                loss.backward()

                optimizer.step()

            print(f'Epoch: {epoch + 1}: Loss: {epoch_loss / len(self.train_loader)}')

    def eval(self, val_loader: DataLoader):

        '''
        Runs evaluation algorithm, prints validation loss. 

        Inputs:
            val_loader: DataLoader
        '''

        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                predictions = self.model(images)

                loss = self.loss_fn(predictions, labels)
                val_loss += loss.item()
            
            print(f'Validation Loss: {val_loss / len(val_loader)}')

    def save_model(self):
        '''saves pretrained classifier to pretrained_model.pt'''
        torch.save(self.model.state_dict(), 'model.pt')
        print('Saved detection model to model.pt')