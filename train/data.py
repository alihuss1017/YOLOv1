from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import os

class BuildTrainDataset(Dataset):

    ''' 
    Builds detection-tasked dataset.

    Inputs:
        path: str

    Outputs: 
        image: torch.Tensor shape = (3, 448, 448)
        label: torch.Tensor
                shape = (5, )
                contents = [class_idx, x_center, y_center, w, h]
    '''

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
                    contents = [int(content) for content in contents]
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
        x_min, y_min, x_max, y_max, class_idx = self.labels[idx]

        # find center coords and w, h of bounding box normalized relative to entire image
        x_center = (x_min + x_max) / 2
        y_center = (y_max + y_min) / 2
        w = x_max - x_min
        h = y_max - y_min

        x_center /= 64
        y_center /= 64
        h /= 64
        w /= 64
        
        label = torch.tensor([class_idx, x_center, y_center, w, h]) 
        return image, label
    
