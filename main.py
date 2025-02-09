import torchvision.transforms as T
from torch.utils.data import DataLoader

from Loader import NumberDataset
from Model import Model
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
model_types = ['EasyOCR', 'ResNet', 'CRNN', 'CNN']
model_type = model_types[0]

if model_type != 'EasyOCR':
    # разбитие исходного тренировочного датасета на тренировочный и валидационный
    train_dataset = NumberDataset("data/train.csv", "data/imgs/", char2idx, transform=transform, mode='train_train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)

    val_dataset = NumberDataset("data/train.csv", "data/imgs/", char2idx, transform=transform, mode='train_val')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=val_dataset.collate_fn)

    # выбор созданной модели
    model = Model(char2idx, model_type, 0.005)

    # обучение модели
    model.train(train_dataloader, idx2char, epochs=20)

    # тестирование модели
    cer_score, accuracy = model.evaluate(val_loader, idx2char)

    # загрузка валидационного и тестового датасета
    val_dataset = NumberDataset("data/val.csv", "data/imgs/", char2idx, transform=transform, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=val_dataset.collate_fn)

    test_dataset = NumberDataset("data/test.csv", "data/imgs/", char2idx, transform=transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=test_dataset.collate_fn)

    # выдача предсказаний
    val, test = model.predict()
else:
    dataset = NumberDataset("data/train.csv", "data/imgs/", char2idx, transform=transform, mode='train')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

    model = EasyOCR()
    pred = []
    true = []
    for image, label in dataset.samples:
        num = model.recognize_number(image)
        pred.append(num)
        true.append(label)

    cer_score, accuracy = model.calculate_metrics(pred, true, True)

# вывод точности предсказаний согласно целевым метрикам
print(f'CER: {cer_score:.4f}, Accuracy: {accuracy:.4f}')
