import os
import math
import torch
import random
import requests
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class Constants:

    seed = 42
    batch_size = 32
    lr = 1e-3
    default_author = "openai"
    default_name_hf = "clip-vit-base-patch32"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_device(device_no: int):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(device_no))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
        
    return device