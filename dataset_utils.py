from configs import *
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer

class CLIPDataset():
    def __init__(self, list_image_path: list, list_txt: list):
        self.image_path = list_image_path
        self.title = list_txt

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        title = self.title[idx]
        return image, title


def collate_fn(batch):
    return {
        'image': [x[0] for x in batch],
        'title': [x[1] for x in batch]
    }

def preprocess_loader(loader, concepts: list):
    preprocessed_batches = []
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    for batch in tqdm(loader):
        preprocessed_batch = preprocess_batch(batch, processor, concepts)
        preprocessed_batches.append(preprocessed_batch)
    return preprocessed_batches

def preprocess_batch(batch, processor, concepts: list):
    return processor(text=concepts, images=batch['image'], return_tensors="pt", padding=True), batch['title']