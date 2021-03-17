import torchvision


def load_dataset(dataset_path):
    return torchvision.datasets.ImageFolder(dataset_path)
