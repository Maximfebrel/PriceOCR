from sklearn.model_selection import train_test_split
import editdistance
from CNN import CNNCTCModel
from Loader import DataPreprocessor
import tensorflow as tf


class PriceRecognizer:
    def __init__(self, train_csv_path, val_csv_path, test_csv_path, image_dir, image_size=(64, 64), max_label_length=10):
        """
        Инициализация класса для распознавания цен.

        :param train_csv_path: Путь к train.csv
        :param test_csv_path: Путь к test.csv
        :param image_dir: Путь к директории с изображениями
        :param image_size: Размер изображений после ресайза
        :param max_label_length: Максимальная длина метки
        """
        self.data_preprocessor = DataPreprocessor(train_csv_path, val_csv_path, test_csv_path, image_dir, image_size)
        self.X_train, self.y_train, self.X_test, self.X_val = None, None, None, None
        self.charset = list("0123456789.")  # Символы в ценах
        self.max_label_length = max_label_length
        self.cnn_ctc_model = None

    def run(self, epochs=100, batch_size=32):
        """
        Запуск процесса обучения и оценки модели.

        :param epochs: Количество эпох
        :param batch_size: Размер батча
        """
        # Загрузка данных
        self.X_train, self.y_train, self.X_val, self.X_test = self.data_preprocessor.load_data()

        # Создание и обучение модели
        self.cnn_ctc_model = CNNCTCModel(input_shape=self.X_train.shape[1:], max_label_length=self.max_label_length,
                                         charset=self.charset)

        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

        self.max_label_length = max(len(label) for label in y_train + y_val) + 1
        self.cnn_ctc_model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # Оценка модели
        self.evaluate(X_val, y_val)
        self.predict()

        # object detection
        # faster rcnn

    def evaluate(self, X_val, y_val):
        """
        Оценка модели на тестовых данных.

        :param X_val: Валидационные изображения
        :param y_val: Валидационные метки
        """
        # Получение предсказаний
        y_pred = self.cnn_ctc_model.base_model.predict(tf.convert_to_tensor(X_val, dtype=tf.float32))
        decoded_preds = self.cnn_ctc_model.decode_predictions(y_pred)

        # Вычисление Accuracy и CER
        correct_count = 0
        total_cer = 0
        for pred, true in zip(decoded_preds, y_val):
            if pred == true:
                correct_count += 1
            total_cer += self.calculate_cer(pred, true)

        accuracy = correct_count / len(y_val)
        cer = total_cer / len(y_val)

        print(f"Test Accuracy: {accuracy:.4f}, Test CER: {cer:.4f}")

    def calculate_cer(self, pred, true):
        """
        Вычисление Character Error Rate.

        :param pred: Предсказанная строка
        :param true: Настоящая строка
        :return: CER
        """
        return editdistance.eval(pred, true) / len(true)

    def predict(self):
        y_pred_val = self.cnn_ctc_model.base_model.predict(self.X_val)
        y_pred_test = self.cnn_ctc_model.base_model.predict(self.X_test)


# Пример использования
if __name__ == "__main__":
    train_csv_path = r"C:\Users\makso\Desktop\PriceOCR\PriceOCR\data\train.csv"
    val_csv_path = r"C:\Users\makso\Desktop\PriceOCR\PriceOCR\data\val.csv"
    test_csv_path = r"C:\Users\makso\Desktop\PriceOCR\PriceOCR\data\test.csv"
    image_dir = r"C:\Users\makso\Desktop\PriceOCR\PriceOCR\data\imgs"

    recognizer = PriceRecognizer(train_csv_path, val_csv_path, test_csv_path, image_dir)
    recognizer.run(epochs=100, batch_size=32)
