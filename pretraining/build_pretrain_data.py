from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class PreTrainDataset(Dataset):
    '''
    Creates classification-tasked dataset.
    
    Inputs:
        root: str
    '''

    def __init__(self, root: str):
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.root = root

        self.dataset =  ImageFolder(root = self.root, 
                        transform = self.transform,
                        is_valid_file = self._file_checker)
        
    def _file_checker(self, file: str) -> bool: 
        '''checks if file is valid format.'''
        return file.lower().endswith(('.jpg', '.jpeg', '.png'))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]

