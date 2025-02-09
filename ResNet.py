import torch.nn as nn
import torch


class ResNetCRNN(nn.Module):
    """Архитектура с использованием предобученного ResNet18 в качестве энкодера"""

    def __init__(self, num_chars: int, hidden_size=256):
        """

        :param num_chars: количество распознаваемых символов
        :param hidden_size: размер скрытого слоя для рекуррентной сети
        """
        super().__init__()
        # Используем предобученный ResNet
        base_model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

        # Модифицируем первый слой для одноканальных изображений
        base_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Убираем последние слои ResNet
        self.encoder = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )

        # Адаптер для RNN
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # RNN часть
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )

        # Классификатор
        self.fc = nn.Linear(hidden_size * 2, num_chars)

    def forward(self, x: torch.Tensor):
        """
        Функция для прямого прохода нейронной сети
        :param x: вход нейронной сети
        :return:
        """
        # Энкодинг изображения
        x = self.encoder(x)  # [batch, 512, h, w]
        x = self.adaptive_pool(x)  # [batch, 512, 1, seq_len]
        x = x.squeeze(2)  # [batch, 512, seq_len]
        x = x.permute(2, 0, 1)  # [seq_len, batch, features]

        # RNN
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


