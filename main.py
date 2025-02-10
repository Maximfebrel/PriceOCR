import torchvision.transforms as T
from torch.utils.data import DataLoader

from Loader import NumberDataset
from Model import Model
from ModelBox import ModelBox
from EasyOCR import EasyOCR


# Символы, которые предполагается распознавать
char2idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, ' ': 10}  # пробел как blank
idx2char = {v: k for k, v in char2idx.items()}

# Предобработка исходных картинок (изменений размера, конвертирование в тензор, нормализация)
transform = T.Compose([
    T.Resize((32, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# выбор вида модели
model_types = ['EasyOCR', 'CRNN', 'CNN', 'ModelBox']
model_type = model_types[3]

if model_type in ['CRNN', 'CNN']:
    # загрузка тренировочного датасета
    train_dataset = NumberDataset("data/train.csv", "data/imgs/", char2idx, transform=transform, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)

    # выбор созданной модели
    model = Model(char2idx, model_type, 0.01)

    # обучение модели
    model.train(train_dataloader, idx2char, epochs=20)

    # загрузка валидационного датасета
    val_dataset = NumberDataset("data/val.csv", "data/imgs/", char2idx, transform=transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=val_dataset.collate_fn)

    # тестирование модели
    cer_score, accuracy = model.evaluate(val_loader, idx2char)

    # загрузка тестового датасета
    test_dataset = NumberDataset("data/test.csv", "data/imgs/", char2idx, transform=transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=test_dataset.collate_fn)

    # выдача предсказаний
    test = model.predict(test_loader, idx2char)
    test_dataset.to_csv(test, model_type)
elif model_type == 'ModelBox':
    # загрузка тренировочного датасета
    train_dataset = NumberDataset("data/train.csv", "data/imgs/", char2idx, transform=transform, mode='train')
    bounding_box = train_dataset.make_box()
    train_dataloader = DataLoader(bounding_box, batch_size=64)

    # выбор созданной модели
    model = ModelBox(model_type, 0.005)
    model.train(train_dataloader, epochs=10)
else:
    # загрузка валидационного датасета
    dataset = NumberDataset("data/val.csv", "data/imgs/", char2idx, transform=transform, mode='val')

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
    test_dataset = NumberDataset("data/test.csv", "data/imgs/", char2idx, transform=transform, mode='test')
    pred = []
    for image, label in test_dataset.samples:
        num = model.recognize_number(image)
        pred.append(num)
    test_dataset.to_csv(pred, model_type)

# вывод точности предсказаний согласно целевым метрикам
print(f'CER: {cer_score:.4f}, Accuracy: {accuracy:.4f}')
