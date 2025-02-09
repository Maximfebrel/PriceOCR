import torchvision.transforms as T
from torch.utils.data import DataLoader

from Loader import NumberDataset
from Model import Model
from EasyOCR import EasyOCR


# Параметры
char2idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
            '6': 6, '7': 7, '8': 8, '9': 9, ' ': 10}  # пробел как blank
idx2char = {v: k for k, v in char2idx.items()}

transform = T.Compose([
    T.Resize((32, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])


model_type = 'EasyOCR'
# выбор модели
if model_type != 'EasyOCR':
    train_dataset = NumberDataset("data/train.csv", "data/imgs/", char2idx, transform=transform, mode='train_train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)

    val_dataset = NumberDataset("data/train.csv", "data/imgs/", char2idx, transform=transform, mode='train_val')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=val_dataset.collate_fn)

    model_cnn = Model(char2idx, model_type)

    # обучение модели
    model_cnn.train(train_dataloader, idx2char, epochs=20)

    # тестирование модели
    cer_score, accuracy = model_cnn.evaluate(val_loader, idx2char)
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

print(f'CER: {cer_score:.4f}, Accuracy: {accuracy:.4f}')
