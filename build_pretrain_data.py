from torchvision import transforms
from torchvision.datasets import ImageFolder

class PreTrainDataset:
    def __init__(self, root: str):
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.root = root

    def _file_checker(self, file: str):
        return file.lower().endswith(('.jpg', '.jpeg', '.png'))

    def run(self):
        'builds and returns dataset'
        dataset = ImageFolder(root = self.root, 
                        transform = self.transform,
                        is_valid_file = self._file_checker)
        
        return dataset

