import glob
from itertools import chain
import os
import random
import json
import time

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from tqdm.notebook import tqdm
from torchvision.models import resnet18
from torchvision.transforms import ColorJitter, RandomApply

from vit_pytorch.efficient import ViT

import argparse

# Training settings
batch_size = 12
epochs = 3
lr = 3e-5
gamma = 0.7
seed = 42
device = 'cuda'

data_dir = 'train_data'

fonts = [f.split('/')[-1].split('.')[0] for f in glob.glob('ttfs2/*.ttf')]
NUM_CLASSES = len(fonts)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class ResizeFixedHeight(transforms.Resize):
    def __init__(self, fixed_height: int, interpolation: int = Image.BICUBIC):
        self.fixed_height = fixed_height
        self.interpolation = interpolation

    def __call__(self, img):
        width, height = img.size
        new_height = self.fixed_height
        new_width = int(width * (new_height / height))
        return img.resize((new_width, new_height), self.interpolation)


class CustomPadToMinSize(transforms.Pad):
    def __init__(self, min_size: Tuple[int, int], fill=0, padding_mode='constant'):
        self.min_size = min_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        width, height = img.size
        pad_left = max(0, (self.min_size[1] - width) // 2)
        pad_top = max(0, (self.min_size[0] - height) // 2)
        padding = (pad_left, pad_top, self.min_size[1] - width - pad_left, self.min_size[0] - height - pad_top)

        return F.pad(img, padding, self.fill, self.padding_mode)


seed_everything(seed)

train_transforms = transforms.Compose(
    [
        ResizeFixedHeight(80),
        CustomPadToMinSize((224, 224), fill=0, padding_mode='constant'),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.33),
    ]
)

val_transforms = transforms.Compose(
    [
        ResizeFixedHeight(80),
        CustomPadToMinSize((224, 224), fill=0, padding_mode='constant'),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        ResizeFixedHeight(80),
        transforms.Pad((0, 48, 0, 48), fill=0, padding_mode='constant'),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

def main():
    train_list = glob.glob(os.path.join(data_dir,'*.png'))

    labels = [json.load(open(f.replace('.png', '.json')))['font'] for f in train_list]

    train_list, test_list = train_test_split(train_list, test_size=0.1, stratify=labels, random_state=seed)

    print(f"Train Data: {len(train_list)}")
    print(f"Test Data: {len(test_list)}")

    train_data = MyDataset(train_list, transform=train_transforms)
    test_data = MyDataset(test_list, transform=test_transforms)

    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

    print('len(train_data), len(train_loader)', len(train_data), len(train_loader))
    print('len(test_data), len(test_loader)', len(test_data), len(test_loader))

    if args.model == 'vit':
        efficient_transformer = Linformer(
            dim=128,
            seq_len=49+1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
        )
        model = ViT(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=NUM_CLASSES,
            transformer=efficient_transformer,
            channels=3,
        ).to(device)
    elif args.model == 'resnet18':
        model = resnet18(num_classes=NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    iter = 0
    start_time = time.time()  # Start the timer for each epoch

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        iter = 0
        start_time = time.time()  # Start the timer for each epoch

        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.item()
            epoch_loss += loss.item()
            iter += 1

            if iter % 100 == 0:
                # Print the progress of training every 100 iterations
                print(f'Epoch: {epoch+1}/{epochs}, Iteration: {iter}/{len(train_loader)}, Iteration Loss: {loss.item():.4f}, Iteration Acc: {acc.item():.4f}')

            if iter % 10000 == 0:
                with torch.no_grad():
                    epoch_test_accuracy = 0
                    epoch_test_loss = 0
                    for data, label in test_loader:
                        data = data.to(device)
                        label = label.to(device)

                        test_output = model(data)
                        test_loss = criterion(test_output, label)

                        acc = (test_output.argmax(dim=1) == label).float().mean()
                        epoch_test_accuracy += acc.item()
                        epoch_test_loss += test_loss.item()

                avg_test_accuracy = epoch_test_accuracy / len(test_loader)
                avg_test_loss = epoch_test_loss / len(test_loader)
                print(f'Epoch: {epoch+1}/{epochs}, Iteration: {iter}/{len(train_loader)}, Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_accuracy:.4f}')
                torch.save(model.state_dict(), 'models/model_{:02d}.pth'.format(epoch+1))

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_accuracy = epoch_accuracy / len(train_loader)
        end_time = time.time()  # End the timer after each epoch
        print(f'Epoch {epoch+1} completed in {(end_time - start_time):.2f} seconds, Avg Train Loss: {avg_epoch_loss:.4f}, Avg Train Acc: {avg_epoch_accuracy:.4f}')
        scheduler.step()


class MyDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.font_mapping = self.create_font_mapping(file_list)

    def create_font_mapping(self, file_list):
        font_names = set()
        for f in file_list:
            jsn_path = f.replace('.png', '.json')
            with open(jsn_path) as json_file:
                font = json.load(json_file)['font']
                font_names.add(font)
        font_mapping = {font: i for i, font in enumerate(sorted(font_names))}
        return font_mapping
    
    def create_reverse_font_mapping(self):
        return {v: k for k, v in self.font_mapping.items()}

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        jsn_path = img_path.replace('.png', '.json')
        img = Image.open(img_path).convert('RGB')
        img_transformed = self.transform(img)

        with open(jsn_path) as json_file:
            label = json.load(json_file)['font']
        label = self.font_mapping[label]

        return img_transformed, label

def save_preview_images(images, labels, font_mapping, reverse_font_mapping, output_dir='preview_images'):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    n_images = len(images)
    for i, (img, label) in enumerate(zip(images, labels)):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis('off')
        ax.set_title(f"Label: {label} ({reverse_font_mapping[label]})")
        plt.savefig(os.path.join(output_dir, f"preview_image_{i}.png"))
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Font recognition training script')
    parser.add_argument('-m', '--model', type=str, default='vit', choices=['vit', 'resnet18'], help='Model to use for training (default: vit)')
    parser.add_argument('-p', '--preview', action='store_true', help='Preview training data samples')
    args = parser.parse_args()

    if args.preview:
        train_list = glob.glob(os.path.join(data_dir,'*.png'))[:200]
        labels = [json.load(open(f.replace('.png', '.json')))['font'] for f in train_list]
        train_list, test_list = train_test_split(train_list, test_size=0.1, stratify=labels, random_state=seed)
        n_preview_images = 5  # Number of images to preview
        preview_images_list = train_list[:n_preview_images]
        preview_data = MyDataset(preview_images_list, transform=train_transforms)
        images, labels = zip(*[preview_data[i] for i in range(n_preview_images)])
        reverse_font_mapping = preview_data.create_reverse_font_mapping()
        save_preview_images(images, labels, preview_data.font_mapping, reverse_font_mapping=reverse_font_mapping)
    else:
        main()