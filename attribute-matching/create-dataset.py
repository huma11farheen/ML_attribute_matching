from pathlib import Path
from typing import List
from argparse import ArgumentParser
from tqdm import tqdm


from facenet_pytorch import InceptionResnetV1

from utils import get_image_loader

import pickle

import PIL
import pandas as pd
import torch



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=Path)
    parser.add_argument('-p', '--pattern', type=str)
    parser.add_argument('-o', '--output', type=Path)
    return parser.parse_args()


def get_filelist(data_dir:Path, pattern: str):
    return list(data_dir.glob(f'**/{pattern}'))


def main(filelist: List):
    device = torch.device('cpu')
    loader = get_image_loader()
    resnet = InceptionResnetV1(pretrained='vggface2',
                               classify=False,
                               device=device).eval()
    data = []
    embeddings = []
    for filepath in tqdm(filelist):
        image = loader(filepath).to(device)
        with torch.no_grad():
            embedding = resnet(image).cpu()
        data.append({
            'filename' : filepath,
            'embeddings': embedding,
        })
        embeddings.append(embedding)
    data = pd.DataFrame(data)
    embeddings = torch.cat(embeddings, dim=0)
    return {
                'data': data,
                'embeddings': embeddings, #Tensor representation
            }

def save_database(database, output:Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'wb') as f:
        pickle.dump(database, f)

def test_saved_file(database, output):
    with open(output, 'rb') as f:
        reloaded = pickle.load(f)
    print(reloaded)
    print(database)

if __name__ == '__main__':
    args = parse_args()
    filelist = get_filelist(args.data_dir, args.pattern)
    database = main(filelist)
    save_database(database, args.output)
    # test_saved_file(database, args.output)