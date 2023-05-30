import argparse
import logging

import torch

from dataset import NumpyImages
from utils import build_model

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--log-dir', type=str, default='results.log')
parser.add_argument('--model-name', type=str, default=None)
parser.add_argument('--batch-size', type=int, default=200)
args = parser.parse_args()

args.model_name = args.model_name.split(",")

log_file_dir = args.log_dir
logging.basicConfig(filename=log_file_dir, 
                    format = '%(message)s',
                    level=logging.WARNING)

logging.warning(log_file_dir)

def evaluate(model, dataloader):
    n_img, n_correct = 0, 0
    for img, label in dataloader:
        label, img = label.cuda(), img.cuda()
        with torch.no_grad():
            pred = torch.argmax(model(img), dim=1).view(1,-1)
        n_correct += (label != pred.squeeze(0)).sum().item()
        n_img += len(label)
    return round(100. * n_correct / n_img, 2)

dataset = NumpyImages(args.dir)
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=args.batch_size, 
                                         num_workers=4, 
                                         pin_memory=True)

for mname in args.model_name:
    model, data_config = build_model(mname)
    accuracy = evaluate(model, dataloader)
    logging.warning("{}, {}, {}".format(args.dir, mname, accuracy))
