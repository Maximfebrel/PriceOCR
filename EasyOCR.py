import easyocr
import cv2
from jiwer import cer


class EasyOCR:
    def __init__(self):
        # Инициализация ридера (указываем английский язык и только цифры)
        self.reader = easyocr.Reader(['en'])

    def recognize_number(self, image_path):
        # Распознавание текста с ограничением только на цифры
        results = self.reader.readtext(image_path,
                                       allowlist='0123456789',
                                       decoder='greedy',
                                       detail=0)

        # Объединение результатов
        return ''.join(results) if results else ''

    @staticmethod
    def calculate_metrics(preds, targets, mode):
        # CER
        cer_score = cer(targets, preds)

        # Accuracy (полное совпадение)
        correct = sum([1 for p, t in zip(preds, targets) if p == t])
        if mode:
            for i in range(len(preds)):
                print(preds[i], targets[i])
        accuracy = correct / len(targets)

        return cer_score, accuracy

