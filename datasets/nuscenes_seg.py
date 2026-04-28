# datasets/nuscenes_seg.py
import os
from torch.utils.data import Dataset
from PIL import Image
import sys

sys.path.insert(0, '/content/drive/MyDrive/adas_occ_project')
from config import IMG_DIR, MASK_DIR, TRAIN_SPLIT

class NuScenesSegDataset(Dataset):
    def __init__(self, processor, split='train',
                 img_dir=IMG_DIR, mask_dir=MASK_DIR):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.processor = processor

        tokens = sorted([f.replace('.jpg', '')
                         for f in os.listdir(img_dir)
                         if f.endswith('.jpg')])
        split_idx = int(len(tokens) * TRAIN_SPLIT)
        self.tokens = tokens[:split_idx] if split == 'train' \
                      else tokens[split_idx:]
        print(f"[NuScenesSegDataset] {split}: {len(self.tokens)} samples")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        img  = Image.open(f'{self.img_dir}/{token}.jpg').convert('RGB')
        mask = Image.open(f'{self.mask_dir}/{token}.png')
        encoding = self.processor(
            images=img,
            segmentation_maps=mask,
            return_tensors='pt'
        )
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'labels':       encoding['labels'].squeeze(),
        }