from train import BuildTrainDataset, DetectionConvNet, Trainer, BuildLoaders
from pretrain import ClassifierConvNet
from utils.loss import YOLOLoss
import torch

batch_size = 64
num_classes = 200
subset_size = 5000
dataset = BuildTrainDataset('tiny-imagenet-200/train')
builder = BuildLoaders(dataset, batch_size, subset_size)

pretrained_convNet = ClassifierConvNet(num_classes)
convNet = DetectionConvNet(pretrained_convNet)
loss_fn = YOLOLoss()

train_loader, val_loader = builder.run()

def load_images_and_labels():
    images, labels = next(iter(train_loader))
    return images, labels 

def generate_predictions(images):
    return convNet(images)

def test_loader():
    images, labels = load_images_and_labels()
    assert images.shape == (batch_size, 3, 448, 448)
    assert labels.shape == (batch_size, 5)

def test_model():
    images, _ = load_images_and_labels()
    assert convNet(images).shape == (batch_size, 7, 7, 210)

def test_YOLO():
    images, labels = load_images_and_labels()
    predictions = generate_predictions(images)
    loss = loss_fn(predictions, labels)
    loss_val = loss.item()
    assert isinstance(loss_val, float)