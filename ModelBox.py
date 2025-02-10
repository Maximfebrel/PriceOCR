import torch
import torch.nn as nn
from jiwer import cer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from CNN import CNNBox


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
        self.model = CNNBox(10).to(self.device)
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

                # представление меток класса в виде для лосса
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
                loss = self.criterion(outputs, targ.type(torch.float))

                # обратный проход
                loss.backward()
                self.optimizer.step()

                all_preds.extend(torch.max(outputs.data, 1)[1])
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
        plt.ylabel('Cross Entropy')
        plt.xlabel('Epoch')
        plt.savefig('result/loss.png')

    def evaluate(self, dataloader: DataLoader, target_len: list) -> tuple:
        """
        Функция для оценки качества построенной архитектуры
        :param dataloader: объект, реализующий эффективную передачу по батчам
        :param target_len: количество символов в изображении
        :return: целевые метрики обученной модели
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                # прямой проход
                outputs = self.model(images[None, :].permute(1, 0, 2, 3).type(torch.float))

                # приводим цифры к формату для кодировки последовательностей
                outputs = torch.max(outputs.data, 1)[1].detach().numpy()
                all_preds.append(int(outputs))
                all_targets.append(int(targets))

            all_preds = self.code_sequence(all_preds, target_len)
            all_targets = self.code_sequence(all_targets, target_len)

            indexes_to_delete = []
            # удаляем нераспознанные символы
            for i in range(len(all_preds)):
                if all_preds[i] == '':
                    indexes_to_delete.append(i)

            all_preds_clear = []
            all_targets_clear = []
            for i in range(len(all_preds)):
                if i not in indexes_to_delete:
                    all_preds_clear.append(all_preds[i])
                    all_targets_clear.append(all_targets[i])

        # вычисление целевых метрик
        cer_score, accuracy = self.calculate_metrics(all_preds_clear, all_targets_clear, False)
        return cer_score, accuracy

    @staticmethod
    def code_sequence(sequence, sequence_lengths):
        """
        Функция для кодировки последовательностей
        :param sequence: последовательность цифр
        :param sequence_lengths: длина последовательности символов в числе
        :return: последовательность чисел
        """
        sum_len = 0
        target = []
        for target_len in sequence_lengths:
            str_target = ''
            for i in range(sum_len, sum_len + target_len):
                str_target += str_target.join(str(sequence[i]))
            target.append(str_target)
            sum_len = target_len
        return target

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

    def predict(self, dataloader: DataLoader, target_len: list) -> list:
        """
        Функция для осуществления предсказаний
        :param dataloader: объект, реализующий эффективную передачу по батчам
        :param target_len: количество символов в изображении
        :return: числа с картинок
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                # прямой проход
                outputs = self.model(images[None, :].permute(1, 0, 2, 3).type(torch.float))

                outputs = torch.max(outputs.data, 1)[1].detach().numpy()
                all_preds.append(int(outputs))

            all_preds = self.code_sequence(all_preds, target_len)

        return all_preds
