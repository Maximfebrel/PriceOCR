import os
import numpy as np
import pandas as pd
import keras


class DataPreprocessor:
    def __init__(self, train_csv_path, val_csv_path, test_csv_path, image_dir, image_size=(64, 64)):
        """
        Инициализация класса для обработки данных.

        :param train_csv_path: Путь к train.csv
        :param val_csv_path: Путь к val.csv
        :param test_csv_path: Путь к test.csv
        :param image_dir: Путь к директории с изображениями
        :param image_size: Размер изображений после ресайза (ширина, высота)
        """
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.image_dir = image_dir
        self.im_size = image_size

    def load_data(self):
        """
        Загрузка данных из CSV-файлов и преобразование их в формат, подходящий для обучения.

        :return: X_train, y_train, X_val, X_test
        """
        # Загрузка данных из CSV
        train_df = pd.read_csv(self.train_csv_path)
        val_df = pd.read_csv(self.val_csv_path)
        test_df = pd.read_csv(self.test_csv_path)

        # Преобразование путей к полным путям к изображениям
        train_df['img_name'] = train_df['img_name'].apply(lambda x: os.path.join(self.image_dir, x))
        val_df = val_df['img_name'].apply(lambda x: os.path.join(self.image_dir, x))
        test_df['img_name'] = test_df['img_name'].apply(lambda x: os.path.join(self.image_dir, x))

        # Загрузка изображений и их меток
        X_train, y_train = self._load_images_and_labels(train_df)
        X_val, _ = self._load_images_and_labels(val_df)
        X_test, _ = self._load_images_and_labels(test_df)

        return X_train, y_train, X_val, X_test

    def _load_images_and_labels(self, df):
        """
        Внутренний метод для загрузки изображений и меток.

        :param df: DataFrame с путями к изображениям и метками
        :return: Массив изображений и массив меток
        """
        images = []
        labels = []
        if isinstance(df, pd.DataFrame):
            for _, row in df.iterrows():
                image_path = row['img_name']
                if 'text' in df.columns:
                    label = str(row['text'])  # Преобразуем цену в строку
                    labels.append(label)

                # Чтение изображения и изменение размера
                image = keras.preprocessing.image.load_img(image_path, target_size=self.im_size, color_mode='grayscale')
                image = keras.preprocessing.image.img_to_array(image) / 255.0  # Нормализация значений пикселей
                images.append(image)
        else:
            for image_path in list(df):
                image = keras.preprocessing.image.load_img(image_path, target_size=self.im_size, color_mode='grayscale')
                image = keras.preprocessing.image.img_to_array(image) / 255.0  # Нормализация значений пикселей
                images.append(image)

        return np.array(images), labels
