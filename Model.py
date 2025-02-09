import torch
import torch.nn as nn
from jiwer import cer
import matplotlib.pyplot as plt

from CNN import CRNN
from ResNet import ResNetCRNN


class Model:
    def __init__(self, char2idx, model_type):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        match model_type:
            case 'CRNN':
                self.model = CRNN(num_chars=len(char2idx)).to(self.device)
            case 'ResNet':
                self.model = ResNetCRNN(num_chars=len(char2idx)).to(self.device)
        self.criterion = nn.CTCLoss(blank=10)  # blank символ имеет индекс 10
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

    def train(self, dataloader, idx2char, epochs=10):
        self.model.train()
        total_loss_list = []

        for epoch in range(epochs):
            total_loss = 0
            all_preds = []
            all_targets = []

            for batch_idx, (images, targets, target_lengths) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                input_lengths = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                )

                loss = self.criterion(
                    outputs.log_softmax(2),
                    targets,
                    input_lengths,
                    target_lengths
                )

                loss.backward()
                self.optimizer.step()

                preds = self.decode_greedy(outputs, idx2char)

                targets = targets.detach().numpy()
                target_lengths = target_lengths.detach().numpy()

                sum_len = 0
                target = []
                for target_len in target_lengths:
                    str_target = ''
                    for i in range(sum_len, sum_len + target_len):
                        str_target += str_target.join(idx2char[targets[i]])
                    target.append(str_target)
                    sum_len = target_len

                all_preds.extend(preds)
                all_targets.extend(target)

                total_loss += loss.item()

            cer_score, accuracy = self.calculate_metrics(all_preds, all_targets, False)

            print(f'Epoch {epoch + 1} Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}, CER: {cer_score:.4f}')
            total_loss_list.append(total_loss / len(dataloader))
        plt.plot(total_loss_list)
        plt.show()

    def evaluate(self, dataloader, idx2char):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets, target_lengths in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)

                preds = self.decode_greedy(outputs, idx2char)

                targets = targets.detach().numpy()
                target_lengths = target_lengths.detach().numpy()

                sum_len = 0
                target = []
                for target_len in target_lengths:
                    str_target = ''
                    for i in range(sum_len, sum_len + target_len):
                        str_target += str_target.join(idx2char[targets[i]])
                    target.append(str_target)
                    sum_len = target_len

                all_preds.extend(preds)
                all_targets.extend(target)

        cer_score, accuracy = self.calculate_metrics(all_preds, all_targets, True)
        return cer_score, accuracy

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

    @staticmethod
    def decode_greedy(output, idx2char):
        # output: [seq_len, batch, num_classes]
        output = output.permute(1, 0, 2)  # [batch, seq_len, num_classes]
        _, max_indices = torch.max(output, 2)

        decoded_strings = []
        for batch in max_indices:
            chars = []
            prev_char = None
            for idx in batch:
                char = idx2char[idx.item()]
                if char != prev_char and char != ' ':
                    chars.append(char)
                prev_char = char
            decoded_strings.append(''.join(chars))
        return decoded_strings

    @staticmethod
    def decode_beam(output, idx2char, beam_width=3):
        # output: [seq_len, num_classes]
        sequences = [[[], 0.0]]
        for step in output:
            all_candidates = []
            for seq, score in sequences:
                for idx, log_prob in enumerate(step):
                    if idx == 10:  # пропускаем blank
                        continue
                    char = idx2char[idx]
                    new_seq = seq.copy()
                    if len(new_seq) == 0 or new_seq[-1] != char:
                        new_seq.append(char)
                    candidate = [new_seq, score + log_prob]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]
        return ''.join(sequences[0][0])

