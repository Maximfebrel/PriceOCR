import tensorflow as tf
from keras import layers, models
import numpy as np


class CTCLossLayer(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(CTCLossLayer, self).__init__(**kwargs)
        self.num_classes = num_classes

    def call(self, inputs):
        y_true, y_pred, input_length, label_length = inputs

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Вычисление CTC Loss
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=self.num_classes - 1
        )

        # Добавляем loss в модель
        self.add_loss(tf.reduce_mean(loss))

        # Возвращаем фиктивное значение
        return tf.zeros_like(loss)  # Фиктивный тензор для удовлетворения Keras


class CNNCTCModel:
    def __init__(self, input_shape, max_label_length, charset):
        """
        Инициализация CNN+CTC модели.

        :param input_shape: Форма входных данных (ширина, высота, каналы)
        :param max_label_length: Максимальная длина метки
        :param charset: Словарь символов
        """
        self.input_shape = input_shape
        self.max_label_length = max_label_length
        self.charset = charset
        self.num_classes = len(charset) + 1  # +1 для "blank" символа
        self.model, self.base_model = self.build_model()

    def build_model(self):
        """
        Создание архитектуры CNN+CTC.

        :return: Модель для обучения (с CTC Loss) и базовая модель для предсказаний
        """
        # Входное изображение
        inputs = layers.Input(shape=self.input_shape, name="input_image")

        # Сверточные слои
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Превращаем 2D-изображение в 1D-последовательность
        new_shape = ((self.input_shape[0] // 8), (self.input_shape[1] // 8) * 128)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNN слой для генерации последовательностей
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

        # Выходной слой
        output = layers.Dense(self.num_classes, activation="softmax", name="output")(x)

        # Базовая модель для предсказаний
        base_model = models.Model(inputs=inputs, outputs=output, name="Base_OCR_Model")

        # Создаем входы для CTC Loss
        labels = layers.Input(name="labels", shape=(self.max_label_length,), dtype="float32")
        input_length = layers.Input(name="input_length", shape=(1,), dtype="int32")[:, 0]
        label_length = layers.Input(name="label_length", shape=(1,), dtype="int32")[:, 0]

        # Расчет CTC Loss через пользовательский слой
        ctc_loss_output = CTCLossLayer(name="ctc_loss", num_classes=self.num_classes)(
            [labels, output, input_length, label_length])

        # Финальная модель для обучения
        train_model = models.Model(
            inputs=[inputs, labels, input_length, label_length],
            outputs=ctc_loss_output,
            name="OCR_Model"
        )

        return train_model, base_model

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        """
        Обучение модели.

        :param X_train: Обучающие изображения
        :param y_train: Обучающие метки
        :param epochs: Количество эпох
        :param batch_size: Размер батча
        """
        # Подготовка данных для CTC
        train_dataset = self.prepare_dataset(X_train, y_train, batch_size)

        # Компиляция модели
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(optimizer=optimizer)

        # Обучение модели
        history = self.model.fit(train_dataset, epochs=epochs)
        return history

    def prepare_dataset(self, X, y, batch_size):
        """
        Подготовка датасета для CTC.

        :param X: Изображения
        :param y: Метки
        :param batch_size: Размер батча
        :return: TF Dataset
        """

        def generator():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]

                # Подготовка входных данных
                input_length = np.ones(batch_X.shape[0]) * ((self.input_shape[0] // 8) - 2)
                label_length = np.array([min(len(label), self.max_label_length) for label in batch_y])

                # One-hot encoding меток
                batch_y_encoded = np.zeros((batch_X.shape[0], self.max_label_length), dtype=np.int32)
                for j, label in enumerate(batch_y):
                    encoded_label = [self.charset.index(char) for char in label[:self.max_label_length]]
                    batch_y_encoded[j, :len(encoded_label)] = encoded_label

                # Создание словаря с правильными ключами
                data = {
                    "input_image": tf.convert_to_tensor(batch_X, dtype=tf.float32),
                    "labels": tf.convert_to_tensor(batch_y_encoded, dtype=tf.int32),
                    "keras_tensor_12CLONE": tf.convert_to_tensor(input_length.reshape((-1, 1)), dtype=tf.int32),
                    "keras_tensor_13CLONE": tf.convert_to_tensor(label_length, dtype=tf.int32)
                }

                yield data, None

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                {
                    "input_image": tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32),
                    "labels": tf.TensorSpec(shape=(None, self.max_label_length), dtype=tf.int32),
                    "keras_tensor_12CLONE": tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
                    "keras_tensor_13CLONE": tf.TensorSpec(shape=(None,), dtype=tf.int32)
                },
                tf.TensorSpec(shape=(), dtype=tf.float32)  # Пустые целевые значения
            )
        ).prefetch(tf.data.AUTOTUNE)

        return dataset

    def decode_predictions(self, y_pred):
        """
        Декодирование предсказаний.

        :param y_pred: Предсказанные значения
        :return: Декодированные строки
        """
        decoded_strings = []
        for pred in y_pred:
            decoded = "".join([self.charset[char] for char in np.argmax(pred, axis=1) if char != self.num_classes - 1])
            decoded_strings.append(decoded)
        return decoded_strings
