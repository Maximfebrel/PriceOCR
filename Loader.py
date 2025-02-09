import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


class NumberDataset(Dataset):
    def __init__(self, path_excel, img_dir, char2idx, transform=None, mode='train', img_width=128, img_height=32):
        self.path_excel = path_excel
        self.img_dir = img_dir
        self.transform = transform
        self.char2idx = char2idx
        self.mode = mode
        self.img_width = img_width
        self.img_height = img_height

        self.samples = []

        self._load_data()

    def _load_data(self):
        match self.mode:
            case 'train':
                train_data = pd.read_csv(self.path_excel)

                for _, row in train_data.iterrows():
                    label = str(int(row['text']))
                    self.samples.append((self.img_dir + row['img_name'], label))
            case 'train_train':
                train_data = pd.read_csv(self.path_excel)

                train_data, _ = train_test_split(train_data, test_size=0.1, random_state=24)

                for _, row in train_data.iterrows():
                    label = str(int(row['text']))
                    self.samples.append((self.img_dir+row['img_name'], label))
            case 'train_val':
                train_data = pd.read_csv(self.path_excel)

                _, val_data = train_test_split(train_data, test_size=0.1, random_state=24)

                for _, row in val_data.iterrows():
                    label = str(int(row['text']))
                    self.samples.append((self.img_dir+row['img_name'], label))
            case 'val':
                val_data = pd.read_csv(self.path_excel)
            case 'test':
                test_data = pd.read_csv(self.path_excel)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Загрузка и проверка изображения
            image = Image.open(img_path).convert('L')  # Конвертация в градации серого

            # Применение трансформаций с проверкой размера
            if self.transform:
                image = self.transform(image)
                if image.size() != (1, self.img_height, self.img_width):
                    raise ValueError(f"Invalid image size after transform: {image.size()}")

            # Преобразование метки в тензор
            target = [self.char2idx[c] for c in label]
            return {
                'image': image,
                'target': torch.tensor(target, dtype=torch.long),
                'target_length': torch.tensor([len(target)], dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return self.__getitem__((idx + 1) % self.__len__())

    @staticmethod
    def collate_fn(batch):
        """Кастомная функция для объединения примеров в батч"""
        images = []
        targets = []
        target_lengths = []

        for item in batch:
            images.append(item['image'])
            targets.append(item['target'])
            target_lengths.append(item['target_length'])

        return [torch.stack(images), torch.cat(targets), torch.cat(target_lengths)]
