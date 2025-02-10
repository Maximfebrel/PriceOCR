import torch
import torch.nn as nn
from jiwer import cer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2
from sklearn.metrics import accuracy_score

from architecture.CNN import CNN


class ModelBox:
    """Функция для выбора и обучения используемой архитектуры нейронной сети"""

    def __init__(self, model_type: str, lr: float):
        """

        :param model_type: обучаемая архитектура (CRNN, CNN, ResNet18)
        :param lr: скорость обучения
        """
        self.model_type = model_type
        # выбираем архитектуру для использования графического процессора для обучения
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # выбор архитектуры
        self.model = CNN(10).to(self.device)
        # выбор лосса, используем лосс кросс-энтропия
        self.criterion = nn.CrossEntropyLoss()
        # выбор оптимизитора
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataloader: DataLoader, epochs=10):
        """

        :param dataloader: объект, реализующий эффективную передачу по батчам
        :param epochs: количество используемых эпох
        :return: обученная модель, лосс
        """
        self.model.train()
        total_loss_list = []

        for epoch in range(epochs):
            total_loss = 0
            all_preds = []
            all_targets = []

            for batch_idx, (images, target) in enumerate(dataloader):
                images = images.to(self.device)
                targets = target.to(self.device)

                self.optimizer.zero_grad()

                # прямой проход
                outputs = self.model(images[None, :].permute(1, 0, 2, 3).type(torch.float))

                targets = targets.detach().numpy()
                targ = []
                for jdx in range(len(targets)):
                    _ = []
                    for idx in range(10):
                        if targets[jdx] == idx:
                            _.append(1)
                        else:
                            _.append(0)
                    targ.append(_)

                targ = torch.tensor(targ)

                # вычисление лосса
                loss = self.criterion(outputs.mean(0, keepdim=True)[0], targ.type(torch.float))

                # обратный проход
                loss.backward()
                self.optimizer.step()

                all_preds.extend(torch.max(outputs.mean(0, keepdim=True)[0].data, 1)[1])
                all_targets.extend(target)

                total_loss += loss.item()

            # вычисление целевых метрик метрик для отслеживания процесса обучения
            accuracy = accuracy_score(all_preds, all_targets)

            # вывод лосса и целевых метрик для отслеживания процесса обучения
            print(f'Epoch {epoch + 1} Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}')
            total_loss_list.append(total_loss / len(dataloader))
        # сохранение графика лосса
        plt.plot(total_loss_list)
        plt.title(self.model_type)
        plt.ylabel('CTC')
        plt.xlabel('Epoch')
        plt.savefig('result/loss.png')

    @staticmethod
    def detect_digits(image_path):
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

        imgs = []
        # Отрисовка bounding boxes
        for (x, y, w, h) in digit_boxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)




    def evaluate(self, dataloader: DataLoader, idx2char: dict) -> tuple:
        """
        Функция для оценки качества построенной архитектуры
        :param dataloader: объект, реализующий эффективную передачу по батчам
        :param idx2char: символы, которые распознаются на картинках
        :return: целевые метрики обученной модели
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets, target_lengths in dataloader:
                images = images.to(self.device)
                # прямой проход
                outputs = self.model(images)

                # декодировка выхода нейронной сети

                targets = targets.detach().numpy()
                target_lengths = target_lengths.detach().numpy()

                # комбинирование набора цифр в исходные числа
                sum_len = 0
                target = []
                for target_len in target_lengths:
                    str_target = ''
                    for i in range(sum_len, sum_len + target_len):
                        str_target += str_target.join(idx2char[targets[i]])
                    target.append(str_target)
                    sum_len = target_len

                all_preds.extend(preds)
                all_targets.extend(target)

        # вычисление целевых метрик
        cer_score, accuracy = self.calculate_metrics(all_preds, all_targets, False)
        return cer_score, accuracy

    @staticmethod
    def calculate_metrics(preds: list, targets: list, mode: bool) -> tuple:
        """
        Функция для расчета метрик
        :param preds: прогнозные значения после выхода нейронной сети
        :param targets: истинные значения целевой переменной
        :param mode: параметр, отвечающий
        :return: целевые метрики
        """
        # Вычисление метрики CER
        cer_score = cer(targets, preds)

        # Accuracy (полное совпадение)
        correct = sum([1 for p, t in zip(preds, targets) if p == t])
        if mode:
            for i in range(len(preds)):
                print(preds[i], targets[i])
        accuracy = correct / len(targets)

        return cer_score, accuracy

    def predict(self, dataloader: DataLoader, idx2char: dict) -> list:
        """
        Функция для осуществления предсказаний
        :param dataloader: объект, реализующий эффективную передачу по батчам
        :param idx2char: символы, которые распознаются на картинках
        :return: числа с картинок
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for images, targets, target_lengths in dataloader:
                images = images.to(self.device)
                # прямой проход
                outputs = self.model(images)

                # декодировка выхода нейронной сети
                preds = self.decode_greedy(outputs, idx2char)
                all_preds.extend(preds)

        return all_preds
