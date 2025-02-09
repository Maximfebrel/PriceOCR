import torch
import torch.nn as nn


class CNN(nn.Module):
    """Класс, реализующий архитектуру сверточной нейронной сети"""

    def __init__(self, num_chars: int):
        """

        :param num_chars: количество распознаваемых символов
        """
        super(CNN, self).__init__()
        # сверточная сеть (Conv2D - сверточный слой, ReLU - функция активации, MaxPool2d - макспулинг,
        # BatchNorm2d - батч норм для стабильности сети
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
        )

        # полносвязный слой
        self.fc = nn.Linear(1024, num_chars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Функция для прямого прохода нейронной сети
        :param x: вход нейронной сети
        :return: тензор, содержащий в себе информацию о содержащейся на картинке информации
        """
        # проход через сверточные слои
        x = self.cnn(x)
        # ресайз
        x = x.permute(3, 0, 2, 1)  # [высота изображения, размер батча, каналы, признаки]
        x = x.reshape(x.size(0), x.size(1), -1)  # [высота изображения, размер батча, признаки]
        # проход через полносвязные слои
        x = self.fc(x)
        return x


class CRNN(nn.Module):
    """Класс, реализующий архитектуру сверточной нейронной сети с учетом последовательности цифр"""

    def __init__(self, num_chars: int, hidden_size=256):
        """

        :param num_chars: количество распознаваемых символов
        :param hidden_size: размер скрытого слоя рекуррентной сети
        """
        super(CRNN, self).__init__()
        # сверточная сеть (Conv2D - сверточный слой, ReLU - функция активации, MaxPool2d - макспулинг,
        # BatchNorm2d - батч норм для стабильности сети
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        # рекуррентный слой
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_size, bidirectional=True, num_layers=2)
        # полносвязный слой
        self.fc = nn.Linear(1024, num_chars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Функция для прямого прохода нейронной сети
        :param x: вход нейронной сети
        :return: тензор, содержащий в себе информацию о содержащейся на картинке информации
        """
        # проход через сверточные слои
        x = self.cnn(x)
        # ресайз
        x = x.permute(3, 0, 2, 1)  # [высота изображения, размер батча, каналы, признаки]
        x = x.reshape(x.size(0), x.size(1), -1)  # [высота изображения, размер батча, признаки]
        # проход через рекуррентный слой
        x, _ = self.lstm(x)
        # проход через полносвязный слой
        x = self.fc(x)
        return x

