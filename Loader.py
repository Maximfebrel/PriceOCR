import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import cv2


class NumberDataset(Dataset):
    """Класс для загрузки исходных данных и их трансформации"""

    def __init__(self, path_excel: [str, Path],
                 img_dir: [str, Path],
                 char2idx: dict,
                 transform=None,
                 mode='train',
                 img_width=128, img_height=32):
        """

        :param path_excel: путь к загруженному excel
        :param img_dir: путь до папки с изображениями
        :param char2idx: символы, которые необходимо распознать
        :param transform: трансформация исходных картинок
        :param mode: какой конкретно датасет загружаем (train, val, test)
        :param img_width: ширина изображения, которую необходимо получить после ресайза
        :param img_height: высота изображения, которую необходимо получить после ресайза
        """
        self.path_excel = path_excel
        self.img_dir = img_dir
        self.transform = transform
        self.char2idx = char2idx
        self.mode = mode
        self.img_width = img_width
        self.img_height = img_height

        self.samples = []
        self.test_data = None

        self._load_data()

    def _load_data(self):
        """
        Функция для загрузки данных
        :return: данные, которые передаются в DataLoader
        """
        if self.mode in ['train', 'val']:
            # загрузка размеченного датасета
            train_data = pd.read_csv(self.path_excel)

            # приводим исходные данные к формате (картинка, метка)
            for _, row in train_data.iterrows():
                label = str(int(row['text']))
                self.samples.append((self.img_dir + row['img_name'], label))
        else:
            # загрузка неразмеченного датасета
            self.test_data = pd.read_csv(self.path_excel)

            for img in list(self.test_data['img_name']):
                label = '0'
                self.samples.append((self.img_dir + img, label))

    def make_box(self):
        """
        Функция для разделения числа на цифры (разбиение по картинке)
        :return: цифры для обучения модели
        """
        bounding_samples = []
        target_len = []
        for image_path, label in self.samples:
            # Загрузка изображения
            img = cv2.imread(image_path)
            if img is None:
                print("Ошибка загрузки изображения")
                return
            # Предварительная обработка изображения
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Поиск контуров
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Фильтрация контуров и создание bounding boxes
            digit_boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                aspect_ratio = w / float(h)
                # Фильтр по размеру и пропорциям
                if area > 100 and 0.2 < aspect_ratio < 1.0:
                    digit_boxes.append((x, y, w, h))
            # Сортировка bounding boxes слева направо
            digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

            try:
                i = 0
                target_len.append(len(digit_boxes))
                # Отрисовка bounding boxes
                for (x, y, w, h) in digit_boxes:
                    # приведение картинок к единому размеру
                    resized_image = cv2.resize(img[y:y + h, x: x + w], (32, 32))
                    # Преобразование изображения в одноканальное (черно-белое)
                    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                    bounding_samples.append((gray_image, int(label[i])))

            except Exception as e:
                print(f"Error devide {str(e)}")
        return bounding_samples, target_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        """
        Функция для загрузки данных в DataLoader, который разбивает их на батчи
        :param idx: индекс загруженного изображения
        :return: {картинка, метка, количество цифр в числе}
        """
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
    def collate_fn(batch) -> list:
        """Кастомная функция для объединения примеров в батч"""
        images = []
        targets = []
        target_lengths = []

        for item in batch:
            images.append(item['image'])
            targets.append(item['target'])
            target_lengths.append(item['target_length'])

        return [torch.stack(images), torch.cat(targets), torch.cat(target_lengths)]

    def to_csv(self, test: list, model_type: str):
        """
        Функция для занесения определенных изображений в csv
        :param test: определенные числа на изображениях
        :param model_type: модель, при помощи которой это было сделано
        """
        self.test_data['text'] = test
        self.test_data.to_csv(f'result/result_{model_type}.csv')
