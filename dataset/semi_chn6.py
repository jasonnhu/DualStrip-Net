from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random
import itertools
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, cfg, name, root, mode, size=None, id_path=None, nsample=None):
        self.cfg = cfg
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        if self.mode == 'val' or self.mode == 'test':
            img = Image.open(os.path.join(self.root, 'val/images/', id+'_sat.jpg')).convert('RGB')
            mask = np.array(Image.open(os.path.join(self.root, 'val/gt/', id+'_mask.png')).convert('L'))
        else:
            img = Image.open(os.path.join(self.root, 'train/images/', id+'_sat.jpg')).convert('RGB')
            mask = np.array(Image.open(os.path.join(self.root, 'train/gt/', id+'_mask.png')).convert('L'))
        mask = mask / 255
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.mode == 'val' or self.mode == 'test':
            img, mask = normalize(img, mask)
            return img, mask, id

       
        img, mask = crop(img, mask, self.size, 255)
        img, mask = hflip(img, mask, p=0.5)


        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)


        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        if self.mode == 'train_l':
            return normalize(img, mask)
        else:
            return normalize(img_w), img_s1, img_s2, ignore_mask

    def __len__(self):
        return len(self.ids)
