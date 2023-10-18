import open_clip
from configs import *

class DownloadCLIP:
    def __init__(self,
                 name: str,
                 author: str,
                 device):
        self.name = name
        self.author = author
        self.device = device

    def load(self):
        clip, _, preprocess = open_clip.create_model_and_transforms(self.name, pretrained=self.author, device=self.device)
        return clip, preprocess

class DownloadCLIPhuggingface:
    def __init__(self,
                 ):
        pass