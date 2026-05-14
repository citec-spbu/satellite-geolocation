import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image


def get_transforms(is_train=True):
    if not is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Датасет для SUES-200 (на котором обучалась модель)
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
            if not v_path.exists():
                continue
                
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

        print(f"SUESDataset загружен {split} ({view}): {len(self.images)} фото")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {
            'image': img,
            'label': self.labels[idx],
            'is_drone': self.is_drone[idx],
            'height': self.heights[idx],
            'path': self.images[idx]
        }


# Датасет для тестовых данных
class UniversalDataset(Dataset):
    def __init__(self, root_dir, transform=None, view='all'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Ищем все папки локаций
        for loc_dir in sorted(self.root_dir.rglob('*')):
            if not loc_dir.is_dir():
                continue
            
            # Определяем тип локации
            loc_type = self._extract_location_type(loc_dir)
            if loc_type is None:
                continue
            
            loc_id = loc_dir.name
            
            # Добавляем спутниковый снимок
            sat_path = loc_dir / 'satellite.jpg'
            if sat_path.exists() and view in ['all', 'satellite']:
                self.samples.append({
                    'path': str(sat_path),
                    'location_type': loc_type,
                    'location_id': loc_id,
                    'is_drone': False
                })
            
            # Добавляем дрон-фото
            if view in ['all', 'drone']:
                drone_patterns = ['uav.jpg', 'uav_0.jpg', 'uav_1.jpg', 'uav_2.jpg']
                for pattern in drone_patterns:
                    drone_path = loc_dir / pattern
                    if drone_path.exists():
                        self.samples.append({
                            'path': str(drone_path),
                            'location_type': loc_type,
                            'location_id': loc_id,
                            'is_drone': True
                        })
        
        # Создаём словарь локаций
        self.locations = sorted(list(set([s['location_id'] for s in self.samples])))
        self.loc_to_idx = {loc: i for i, loc in enumerate(self.locations)}
        
        drone_count = sum(1 for s in self.samples if s['is_drone'])
        sat_count = sum(1 for s in self.samples if not s['is_drone'])
        print(f"UniversalDataset: {len(self.samples)} фото, {len(self.locations)} локаций")
        print(f"  Дрон: {drone_count}, Спутников: {sat_count}")
    
    def _extract_location_type(self, loc_dir):
        for part in loc_dir.parts:
            for loc_name in ['Chuanmei', 'Hangdian', 'Jinrong', 'Ligong']:
                if loc_name in part:
                    return loc_name
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = Image.open(sample['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return {
            'image': img,
            'label': self.loc_to_idx[sample['location_id']],
            'is_drone': sample['is_drone'],
            'path': sample['path'],
            'location_type': sample['location_type'],
            'location_id': sample['location_id']
        }
