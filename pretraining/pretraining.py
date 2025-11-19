from pretraining.pretrain_model import ClassifierConvNet
import torch.nn as nn
import torch.optim as optim
import torch

class PreTrainer:

    ''' 
    PreTrainer engine, trains the classifier ConvNet.

    Inputs:
        model: ClassifierConvNet
        lr: float
        device: str
    '''

    def __init__(self, model: ClassifierConvNet, lr: float, device: str):
        self.model = model.to(device)
        self.lr = lr 
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
  
    def train(self, num_epochs: int, train_loader) -> None:

        '''
        Runs training algorithm. Prints loss per epoch.
        
        Inputs:
            train_loader: DataLoader
        '''

        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            for X, y in train_loader:
                optimizer.zero_grad()

                X, y = X.to(self.device), y.to(self.device)

                y_hat = self.model(X)

                loss = self.loss_fn(y_hat, y)
                epoch_loss += loss.item()
                loss.backward()

                optimizer.step()

            print(f'Epoch {epoch+1}: Loss: {epoch_loss / len(train_loader)}')

    def eval(self, val_loader) -> None:

        '''
        Evaluates accuracy on validation dataloader.
        
        Inputs: 
          val_loader: DataLoader
        '''

        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                probs = self.model(X) # (batch_size, num_classes)
                preds = probs.argmax(dim = 1) # (batch_size, )
                
                correct += (preds == y).sum().item()
                total += len(y)

            print(f'Accuracy: {correct / total}%')
        
    def save_model(self) -> None:
        '''saves pretrained classifier to pretrained_model.pt'''
        
        torch.save(self.model.state_dict(), 'pretraining/pretrained_model.pt')
        print("Saved pretrained classifier model to pretrained_model.pt")