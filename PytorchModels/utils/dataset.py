# coding: utf-8

import torch
from torch.utils.data import Dataset, DataLoader


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = [0] * max_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)
        lens.append(len(label[i]))
    return img, pad_label, lens
    

class MyDataset(Dataset):
    def __init__(self, data_file, label_file):
        with open(data_file, 'r') as fr:
            lines = fr.readlines()
            self.data = [line.strip() for line in lines]
        with open(label_file, 'r') as fr:
            labels = fr.readlines(fr)
            self.labels = [label.strip() for label in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = (self.data[idx], self.labels[idx])
        return data

if __name__ == '__main__':
    dataset = MyDataset('data_demo.txt', 'label_demo.txt')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
