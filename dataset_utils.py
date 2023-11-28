from configs import *
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer

class CLIPDataset():
    """
    Class which creates dataset with joint labels and images itself.
    It is convenient to use CLIPDataset with HuggingFace.

    Usage:
        dataset = CLIPDataset(list_image_path=image_paths, list_txt=labels)

    For HuggingFace compatibility:
        hf_dataset = Dataset.from_dict({
            "image_file_path": dataset.image_path,
            "image": [Image.open(image_path) for image_path in dataset.image_path],
            "labels": dataset.label
         })
    """
    def __init__(self, list_image_path, list_txt):
        """
        Args:
            list_image_path: list of paths to images in memory
            list_txt: list of corresponding labels
        """
        self.image_path = list_image_path
        self.label = list_txt

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image = Image.open(image_path)
        label = self.label[idx]
        return {"image_file_path": image_path, "image": image, "labels": label}


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def process_data(example_batch, model_name: str="openai/clip-vit-base-patch32"):
    processor = CLIPProcessor.from_pretrained(model_name)
    inputs = processor.image_processor([x for x in example_batch['image']], return_tensors="pt")
    inputs['labels'] = example_batch['labels']
    return inputs