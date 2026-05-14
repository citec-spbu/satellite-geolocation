import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

def get_transforms(is_train=True):
    if not is_train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transforms.Compose([
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class SUESDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', view='all'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.is_drone = []
        self.heights = []

        loc_range = range(1, 121) if split == 'train' else range(121, 201)

        views = []
        if view in ['all', 'drone']:
            views.append('drone_view_512')
        if view in ['all', 'satellite']:
            views.append('satellite-view')

        for v_type in views:
            v_path = self.root_dir / v_type
            for img_path in v_path.rglob('*'):
                if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                    continue

                loc_num = None
                for part in img_path.parts:
                    if part.isdigit() and len(part) == 4:
                        loc_num = int(part)
                        break
                if loc_num is None or loc_num not in loc_range:
                    continue

                self.images.append(str(img_path))
                self.labels.append(loc_num)

                is_drn = 'drone' in v_type
                self.is_drone.append(is_drn)

                h = 0
                if is_drn:
                    for val in ['150', '200', '250', '300']:
                        if f"/{val}/" in str(img_path) or val in img_path.name:
                            h = int(val)
                            break
                self.heights.append(h)

        print(f"Загружен {split} ({view}): {len(self.images)} фото")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return (
            img,
            self.labels[idx],
            self.is_drone[idx],
            self.heights[idx],
            self.images[idx]
        )
