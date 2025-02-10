import torchvision.transforms as T
from torch.utils.data import DataLoader

from Loader import NumberDataset
from Model import Model
from ModelBox import ModelBox
from EasyOCR import EasyOCR


def run(char2idx: dict,
        idx2char: dict,
        model_types: dict,
        model_type: str,
        path_train: str,
        path_val: str,
        path_test: str,
        imgs: str):
    """
    Функция для запуска кода
    :param model_types: словарь с моделями
    :param imgs: путь к изображениям
    :param char2idx: символы для распознавания
    :param idx2char: инвертированные символы для распознавания
    :param model_type: используемая модель
    :param path_train: путь к тренировочному датасету
    :param path_val: путь к валидационному датасету
    :param path_test: путь к тестовому датасету
    :return: распечатанные значения метрик
    """
    # Предобработка исходных картинок (изменений размера, конвертирование в тензор, нормализация)
    transform = T.Compose([
        T.Resize((32, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    if model_type in ['CRNN', 'CNN']:
        # загрузка тренировочного датасета
        train_dataset = NumberDataset(path_train, imgs, char2idx, transform=transform, mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)

        # выбор созданной модели
        model = Model(char2idx, model_type, model_types[model_type]['lr'])

        # обучение модели
        model.train(train_dataloader, idx2char, epochs=model_types[model_type]['epochs'])

        # загрузка валидационного датасета
        val_dataset = NumberDataset(path_val, imgs, char2idx, transform=transform, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=val_dataset.collate_fn)

        # тестирование модели
        cer_score, accuracy = model.evaluate(val_loader, idx2char)

        # загрузка тестового датасета
        test_dataset = NumberDataset(path_test, imgs, char2idx, transform=transform, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=test_dataset.collate_fn)

        # выдача предсказаний
        test = model.predict(test_loader, idx2char)
        test_dataset.to_csv(test, model_type)
    elif model_type == 'ModelBox':
        # загрузка тренировочного датасета
        train_dataset = NumberDataset(path_train, imgs, char2idx, transform=transform, mode='train')
        bounding_box, _ = train_dataset.make_box()
        train_dataloader = DataLoader(bounding_box, batch_size=64)

        # выбор созданной модели
        model = ModelBox(model_type, model_types[model_type]['lr'])
        model.train(train_dataloader, epochs=model_types[model_type]['epochs'])

        # валидация модели
        val_dataset = NumberDataset(path_val, imgs, char2idx, transform=transform, mode='val')
        bounding_box, target_len = val_dataset.make_box()
        val_dataloader = DataLoader(bounding_box, batch_size=1, shuffle=False)
        cer_score, accuracy = model.evaluate(val_dataloader, target_len)

        # тестирвование модели
        test_dataset = NumberDataset(path_test, imgs, char2idx, transform=transform, mode='test')
        bounding_box, target_len = test_dataset.make_box()
        test_loader = DataLoader(bounding_box, batch_size=1, shuffle=False)
        test = model.predict(test_loader, target_len)
        test_dataset.to_csv(test, model_type)
    else:
        # загрузка валидационного датасета
        dataset = NumberDataset(path_val, imgs, char2idx, transform=transform, mode='val')

        # расчет метрик на предобученной модели
        model = EasyOCR()
        pred = []
        true = []
        for image, label in dataset.samples:
            num = model.recognize_number(image)
            pred.append(num)
            true.append(label)

        cer_score, accuracy = model.calculate_metrics(pred, true, False)

        # выполнение предсказаний на неразмеченном датасете
        test_dataset = NumberDataset(path_test, imgs, char2idx, transform=transform, mode='test')
        pred = []
        for image, label in test_dataset.samples:
            num = model.recognize_number(image)
            pred.append(num)
        test_dataset.to_csv(pred, model_type)

    # вывод точности предсказаний согласно целевым метрикам
    print(f'CER: {cer_score:.4f}, Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    # Символы, которые предполагается распознавать
    CHAR2IDX = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                ' ': 10}  # пробел как blank
    IDX2CHAR = {v: k for k, v in CHAR2IDX.items()}

    PATH_TRAIN = "data/train.csv"
    PATH_VAL = "data/val.csv"
    PATH_TEST = "data/test.csv"
    IMGS = "data/imgs/"

    # выбор вида модели
    MODEL_TYPES = {'EasyOCR': {},
                   'CRNN': {'epochs': 5, 'lr': 0.01},
                   'CNN': {'epochs': 10, 'lr': 0.01},
                   'ModelBox': {'epochs': 10, 'lr': 0.01}
                   }
    MODEL_TYPE = 'CRNN'

    run(CHAR2IDX, IDX2CHAR, MODEL_TYPES, MODEL_TYPE, PATH_TRAIN, PATH_VAL, PATH_TEST, IMGS)
