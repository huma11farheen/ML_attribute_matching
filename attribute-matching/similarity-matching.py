from pathlib import Path
from argparse import ArgumentParser

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from amutils import get_image_loader, load_database

import torch
import matplotlib.pyplot as plt



class Database(object):
    def __init__(self, output):
        self.database = load_database(output)
        self.database['embeddings'] /= (self.database['embeddings']**2).sum(dim=-1, keepdim=True)
        print('Loaded dataset')
        print(self.database['embeddings'].size())
    
    def match(self, target_embeddings):
        numerator = (self.database['embeddings']*target_embeddings).sum(dim=-1)
        denominator = (target_embeddings**2).sum(dim=-1)
        scores = numerator/denominator
        top_matches = torch.topk(scores, k=4, dim=-1, largest=True, sorted=True)
        return top_matches

    def load_image(self, idx):
        return Image.open(self.database['data'].iloc[idx].filename)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-f', '--filepath', type=Path)
    parser.add_argument('-d', '--database', type=Path)
    return parser.parse_args()


def main(filepath, database):
    device = torch.device('cpu')
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=224,
                  margin=0,
                  device=device).eval()

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2',
                               device=device,
                               classify=False).eval()
    image = Image.open(filepath)
    # print(image.size())
    with torch.no_grad():
        face_cropped=mtcnn(image).to(device)
        embedding = resnet(face_cropped.unsqueeze(0)).unsqueeze(1)
        matches = database.match(embedding)
    values, indices = matches 
    fig = plt.figure()
    axs = fig.subplot_mosaic("""
                             AA12
                             AA34
                             """)
    axs['A'].imshow(image)
    for axid, image_id in enumerate(indices[0], start=1):
        ret_image = database.load_image(int(image_id))
        axs[str(axid)].imshow(ret_image)
    fig.savefig('./output.png')


if __name__ == '__main__':
    args = parse_args()
    db = Database(args.database)
    main(args.filepath, db)