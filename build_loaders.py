import torch
from torch.utils.data import Dataset, Subset, random_split, DataLoader

class BuildLoaders:
    def __init__(self, dataset: Dataset, batch_size: int, subset_size: int | None = None,
                 train_split: float = 0.7, val_split: float = 0.3):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.train_split = train_split
        self.val_split = val_split

    def _apply_subset(self):
        if self.subset_size:
            indices = torch.randint(0, len(self.dataset), (self.subset_size, ))
            self.dataset = Subset(self.dataset, indices)
    
    def run(self):
        '''builds training and validation dataloaders.'''
        self._apply_subset()
        train_data, val_data = random_split(self.dataset, [self.train_split, self.val_split])
        train_loader = DataLoader(train_data, self.batch_size, drop_last = True)
        val_loader = DataLoader(val_data, self.batch_size, drop_last = True)

        return train_loader, val_loader

