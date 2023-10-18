from configs import *
from torchvision.datasets import ImageFolder

image_dataset = ImageFolder(root=,
                            transform=
                            )
class TuningDataset(Dataset):
    def __init__(self,
                 root,
                 transform
                 ):
        self.root = root
        self.transform = preprocess
        pass
    #TODO: add classes here because i'd like to import this in setup.py for drawing settings