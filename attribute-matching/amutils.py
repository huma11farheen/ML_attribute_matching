from pathlib import Path

from torchvision import transforms
from PIL import Image

import pickle


def get_image_loader(size=(224,224)):
    loader = transforms.Compose([
                                    transforms.Lambda(lambda filepath: Image.open(filepath)),
                                    transforms.Resize(size) if size is not None else transforms.Lambda(lambda x: x),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.unsqueeze(0)),
    ])
    return loader


def save_database(database, output:Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'wb') as f:
        pickle.dump(database, f)


def load_database(output:Path):
    with open(output, 'rb') as f:
        reloaded = pickle.load(f)
    return reloaded