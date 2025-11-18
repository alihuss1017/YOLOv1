from model import ConvNet
import torch.nn as nn
import torch.optim as optim
import torch

class PreTrainer:
    def __init__(self, model: ConvNet, lr: float, device: str):
        self.model = model.to(device)
        self.lr = lr 
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
  
    def run(self, num_epochs: int, train_loader):
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

    def eval(self, val_loader):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                probs = self.model(X) # (batch_size, num_classes)
                preds = probs.argmax(dim = 1) # (batch_size, )
                
                correct += (preds == y).sum().item()
                total += len(y)

            print(f'Accuracy: {correct / total}%')
        
    def save_model(self):
        torch.save(self.model.state_dict(), 'pretrained_model.pt')