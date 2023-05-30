import argparse
import os
import random

import numpy as np
import torch
from torch.backends import cudnn

from attacks import ILPD
from utils import build_dataset, build_model

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default=None)
parser.add_argument('--data-info-dir', type=str, default=None)
parser.add_argument('--save-dir', type=str, default=None)
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--constraint', type=str, default="linf", choices=["linf", "l2"])
parser.add_argument('--epsilon', type=float, default=8)
parser.add_argument('--step-size', type=float, default=1)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--model-name', type=str, default="tv_resnet50")
parser.add_argument('--force', default=False, action="store_true")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ilpd-N', default=1, type=int)
parser.add_argument('--ilpd-coef', default=0.1, type=float, help="1/gamma")
parser.add_argument('--ilpd-pos', default="layer2.3", type=str)
parser.add_argument('--ilpd-sigma', default=0.05, type=float)

args = parser.parse_args()
if args.constraint == "linf":
    args.epsilon = args.epsilon / 255.
    args.step_size = args.step_size / 255.
print(args)

SEED = args.seed
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True

os.makedirs(args.save_dir, exist_ok=True if args.force else False)

model, data_config = build_model(args.model_name)

dataloader = build_dataset(args, data_config)

# ATTACK
attacker = ILPD(args, source_model = model)
label_ls = []
for ind, (ori_img, label) in enumerate(dataloader):
    label_ls.append(label)
    ori_img, label = ori_img.cuda(), label.cuda()
    img_adv = attacker(args, ori_img, label, verbose=True)
    np.save(os.path.join(args.save_dir, 'batch_{}.npy'.format(ind)), img_adv)
    print(' batch_{}.npy saved'.format(ind))
np.save(os.path.join(args.save_dir, 'labels.npy'), torch.cat(label_ls).numpy())
print('images saved')
