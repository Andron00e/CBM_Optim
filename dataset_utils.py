from configs import *
from torchvision.datasets import ImageFolder

#TODO: add classes here because i'd like to import this in setup.py for drawing settings

class CLIPDataCollator:
    """
    Class which prepares features for CLIP model.
    Works with Hugging Face datasets for Contrastive Learning.
    """
    def __call__(self, features):
        input_ids = torch.stack([feature["input_ids"] for feature in features])
        attention_mask = torch.stack([feature["attention_mask"] for feature in features])
        labels = torch.tensor([feature["labels"] for feature in features])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
