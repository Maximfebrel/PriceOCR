import torch
import torch.nn as nn
from jiwer import cer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from CNN import CNN, CRNN


class Model:
    """Функция для выбора и обучения используемой архитектуры нейронной сети"""

    def __init__(self, char2idx: dict, model_type: str, lr: float):
        """

        :param char2idx: набор распознаваемых символов
        :param model_type: обучаемая архитектура (CRNN, CNN, ResNet18)
        :param lr: скорость обучения
        """
        self.model_type = model_type
        # выбираем архитектуру для использования графического процессора для обучения
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # выбор архитектуры
        match model_type:
            case 'CRNN':
                self.model = CRNN(num_chars=len(char2idx)).to(self.device)
            case 'CNN':
                self.model = CNN(num_chars=len(char2idx)).to(self.device)
        # выбор лосса, используем лосс СTC
        self.criterion = nn.CTCLoss(blank=10)  # blank символ имеет индекс 10
        # выбор оптимизитора
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataloader: DataLoader, idx2char: dict, epochs=10):
        """

        :param dataloader: объект, реализующий эффективную передачу по батчам
        :param idx2char: символы, которые распознаются на картинках
        :param epochs: количество используемых эпох
        :return: обученная модель, лосс
        """
        self.model.train()
        total_loss_list = []

        for epoch in range(epochs):
            total_loss = 0
            all_preds = []
            all_targets = []

            for batch_idx, (images, targets, target_lengths) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                # прямой проход
                outputs = self.model(images)

                # вычисление длины символов на входе
                input_lengths = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                )

                # вычисление лосса
                loss = self.criterion(
                    outputs.log_softmax(2),
                    targets,
                    input_lengths,
                    target_lengths
                )

                # обратный проход
                loss.backward()
                self.optimizer.step()

                # декодировка выхода нейронной сети
                preds = self.decode_greedy(outputs, idx2char)

                # преобразование тензоров в массивы
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

                total_loss += loss.item()

            # вычисление целевых метрик метрик для отслеживания процесса обучения
            cer_score, accuracy = self.calculate_metrics(all_preds, all_targets, False)

            # вывод лосса и целевых метрик для отслеживания процесса обучения
            print(f'Epoch {epoch + 1} Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}, CER: {cer_score:.4f}')
            total_loss_list.append(total_loss / len(dataloader))
        # сохранение графика лосса
        plt.plot(total_loss_list)
        plt.title(self.model_type)
        plt.ylabel('CTC')
        plt.xlabel('Epoch')
        plt.savefig(f'result/loss_{self.model_type}.png')

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
                preds = self.decode_greedy(outputs, idx2char)

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

    @staticmethod
    def decode_greedy(output: torch.Tensor, idx2char: dict) -> list:
        """
        Функция для декодировки выхода нейронной сети (greedy-декодирование)
        :param output: выход нейронной сети
        :param idx2char: распознаваемые символы
        :return: декодированный выход нейронной сети с которым можно работать
        """
        # output: [длина последовательности, размер батча, количество классов]
        output = output.permute(1, 0, 2)  # [размер батча, длина последовательности, количество классов]
        _, max_indices = torch.max(output, 2)

        # декодировка выхода нейронной сети
        decoded_strings = []
        for batch in max_indices:
            chars = []
            prev_char = None
            for idx in batch:
                char = idx2char[idx.item()]
                if char != prev_char and char != ' ':
                    chars.append(char)
                prev_char = char
            decoded_strings.append(''.join(chars))
        return decoded_strings

    @staticmethod
    def decode_beam(output, idx2char, beam_width=3) -> str:
        """
        Функция для декодировки выхода нейронной сети (beam-декодирование)
        :param output: выход нейронной сети
        :param idx2char: распознаваемые символы
        :param beam_width: количество вариантов для рассмотрения
        :return: декодированный выход нейронной сети с которым можно работать
        """
        # output: [длина последовательности, размер батча, количество классов]
        # декодирование выхода нейронной сети
        sequences = [[[], 0.0]]
        for step in output:
            all_candidates = []
            for seq, score in sequences:
                for idx, log_prob in enumerate(step):
                    if idx == 10:  # пропускаем blank
                        continue
                    char = idx2char[idx]
                    new_seq = seq.copy()
                    if len(new_seq) == 0 or new_seq[-1] != char:
                        new_seq.append(char)
                    candidate = [new_seq, score + log_prob]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]
        return ''.join(sequences[0][0])

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
