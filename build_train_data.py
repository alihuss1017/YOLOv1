from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class BuildTrainDataset(Dataset):
    def __init__(self, path: str):

        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])

        self.samples, self.labels = self._samples_and_labels_writer()

    def _samples_and_labels_writer(self):
        samples, labels = [], []

        for idx, folder_path in enumerate(os.listdir(self.path)):
            samples_path = os.path.join(self.path, folder_path, 'images')
            labels_file = os.path.join(self.path, folder_path, 
                                       f'{folder_path}_boxes.txt')
            
            with open(labels_file, 'r') as f:
                for line in f:
                    contents = line.split('\t')[1:]
                    contents[-1] = contents[-1][:contents[-1].index('\n')]
                    contents.append(idx)
                    labels.append(contents)

            for samples_file in os.listdir(samples_path):
                sample_file_path = os.path.join(samples_path, samples_file)
                sample = Image.open(sample_file_path).convert('RGB')
                samples.append(sample)
        
        return samples, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(self.samples[idx])
        return image, self.labels[idx]
    
