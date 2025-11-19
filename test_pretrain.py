from pretraining import PreTrainDataset, ClassifierConvNet, BuildLoaders
import torch

batch_size = 64
num_classes = 200
subset_size = 5000
dataset = PreTrainDataset('tiny-imagenet-200')
builder = BuildLoaders(dataset, batch_size, subset_size)
convNet = ClassifierConvNet(num_classes)

train_loader, val_loader = builder.run()

def load_images_and_labels():
    return next(iter(train_loader))

def test_dataset():
    assert isinstance(len(dataset), int)
    assert dataset[0][0].shape == (3, 224, 224)
    assert isinstance(dataset[0][1], int)

def test_loader_obj_shapes():
    images, labels = load_images_and_labels()
    assert images.shape == (batch_size, 3, 224, 224)
    assert labels.shape == (batch_size, )

def test_model():
    images, _ = load_images_and_labels()
    assert convNet(images).shape == (batch_size, num_classes)