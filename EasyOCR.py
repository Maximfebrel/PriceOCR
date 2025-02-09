import easyocr
from jiwer import cer


class EasyOCR:
    """Класс для распознания изображений при помощи предобученной архитектуры EasyOCR"""
    def __init__(self):
        # Инициализация ридера
        self.reader = easyocr.Reader(['en'])

    def recognize_number(self, image_path: str) -> str:
        """

        :param image_path: путь до изображения
        :return: распознанное число
        """
        # Распознавание текста с ограничением только на цифры
        results = self.reader.readtext(image_path,
                                       allowlist='0123456789',
                                       decoder='greedy',
                                       detail=0)

        # Объединение результатов
        return ''.join(results) if results else ''

    @staticmethod
    def calculate_metrics(preds: list, targets: list, mode: bool) -> tuple:
        """

        :param preds: распознанные при помощи EasyOCR числа
        :param targets: фактические метки чисел
        :param mode: параметр, отвечающий за вывод прогнозных и фактических чисел (True, False)
        :return:
        """
        # CER
        cer_score = cer(targets, preds)

        # Accuracy (полное совпадение)
        correct = sum([1 for p, t in zip(preds, targets) if p == t])
        if mode:
            for i in range(len(preds)):
                print(preds[i], targets[i])
        accuracy = correct / len(targets)

        return cer_score, accuracy

