import os
import csv
import torch
import torchvision
import pandas as pd
import numpy as np
import h5py as h5py
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from google.colab import drive
from google.colab import files
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.parameter import Parameter

size = 12
heigth_start = (64 // 2) - (size // 2)
heigth_end = heigth_start + size + 1
width_start = 2
width_end = -2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def split_data():
    global X_train, y_train, X_validation, y_validation, X_test, x, y, rand, split_index, file
    X_train = x[rand[:split_index]]
    y_train = y[rand[:split_index]]
    X_validation = x[rand[split_index:]]
    y_validation = y[rand[split_index:]]
    X_test = torch.tensor(file['test_dataset'][...]).permute(0, 3, 1, 2).contiguous()


def process_data():
    global X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed, X_train, y_train, X_validation, y_validation, X_test, heigth_start, heigth_end, width_start, width_end
    X_train_processed = rescaling(X_train[..., heigth_start: heigth_end, width_start: width_end])
    y_train_processed = y_train.to(torch.int64)
    X_val_processed = rescaling(X_validation[..., heigth_start: heigth_end, width_start: width_end])
    y_val_processed = y_validation.to(torch.int64)
    X_test_processed = rescaling(X_test[..., heigth_start: heigth_end, width_start: width_end])


def rescaling(tensor):
    return (tensor.to(torch.float32) * 2 / 255) - 1


def descaling(tensor):
    return ((tensor + 1) * 255 / 2).to(torch.uint8)


def plot(tensor):
    if tensor.dtype == torch.float32:
        tensor = descaling(tensor)
    array = tensor.squeeze().numpy()
    plt.imshow(array, cmap='Greys_r')
    plt.show()


def splitting(tensor, num, size=size):
    start_index = (5 - num) * size // 2
    end_index = start_index + (num * size)
    tensor = tensor[..., start_index:end_index]
    return torch.stack(torch.chunk(tensor, num, dim=2))


def train(model, train, validation, iterations=1000, device=device, update=250):
    model.train()
    model.to(device)
    loss = 0
    samples = 0
    correct = 0
    it = iter(train)
    for i in range(iterations):
        try:
            inputs, labels = next(it)
        except StopIteration:
            it = iter(train)
            inputs, labels = next(it)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss += loss.item() * len(inputs)
        correct += (outputs.argmax(-1) == labels).sum().float().item()
        samples += len(inputs)
        if i % update == update - 1:
            model.eval()
            validation_loss = 0
            validation_correct = 0
            num_val = 0
            for inputs, labels in validation:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item() * len(inputs)
                validation_correct += (outputs.argmax(-1) == labels).sum().float()
                num_val += len(inputs)
            print(f'training: accuracy {correct / samples:.4f}, validation: accurancy {validation_correct / num_val:.4f}')
            loss = 0
            correct = 0
            samples = 0
            model.train()
        if i >= iterations:
            break
    print('Finished')


def predict_nums(tensor):
    one_model.eval()
    if tensor.ndim == 3:
        predictions = one_model(tensor.unsqueeze(0).to(device)).argmax(-1).squeeze()
    else:
        predictions = one_model(tensor.to(device)).argmax(-1)
    one_model.train()
    return predictions + 1

# Take the predictions and write into a csv file
def predict(data, device=device, write_to_csv=True):
    second_model.eval()
    predictions = []
    if torch.is_tensor(data):
        tensor = data.to(device)
        nums = predict_nums(tensor)
        for tensor_i, nums_i in zip(tensor, nums):
            x = splitting(tensor_i, nums_i)
            y = second_model(x).argmax(1).tolist()
            while len(y) != 5:
                y.append(10)
            predictions.append(np.asarray(y))
    else:
        correct = 0
        validation = 0
        for i, (inputs, *_) in enumerate(data):
            tensor = inputs.to(device)
            nums = predict_nums(tensor)
            for tensor_i, nums_i in zip(tensor, nums):
                x = splitting(tensor_i, nums_i)
                y = second_model(x).argmax(1).tolist()
                while len(y) != 5:
                    y.append(10)
                predictions.append(np.asarray(y))

    if write_to_csv:
        labels = []
        for pred in predictions:
            a_str = ''
            for i in pred:
                a_str += str(i)
            labels.append(a_str)
        df = pd.DataFrame(enumerate(labels), columns=['Id', 'Label'])
        df['Label'] = df['Label'].astype('str')
        df.to_csv('sample.csv', index=False)
        files.download('sample.csv')
    return torch.tensor(predictions).cpu()


def accuracy(prediction, gt):
    return torch.mean((prediction == gt).sum(1).eq(5).float()).item()


drive.mount('/content/drive')
file = h5py.File('drive/MyDrive/MNIST_synthetic.h5', 'r')

split = 0.7
torch.random.seed = 0
x = torch.tensor(file['train_dataset'][...]).permute(0, 3, 1, 2).contiguous()
y = torch.tensor(file['train_labels'][...])
rand = torch.randperm(len(x))
split_index = int(split * len(x))

X_train = None
y_train = None
X_validation = None
y_validation = None
X_test = None
split_data()
X_train_processed = None
y_train_processed = None
X_val_processed = None
y_val_processed = None
X_test_processed = None
process_data()

X_retrain = X_train_processed
y_retrain = (y_train_processed != 10).sum(axis=1) - 1
X_revalidation = X_val_processed
y_revalidation = (y_val_processed != 10).sum(axis=1) - 1
X_reretrain = []
for i in range(len(X_retrain)):
    X_reretrain.append(splitting(X_retrain[i], y_retrain[i] + 1))
X_reretrain = torch.cat(X_reretrain, 0)
y_reretrain = y_train[y_train != 10].to(torch.long)
X_rerevalidation = []
for i in range(len(X_revalidation)):
    X_rerevalidation.append(splitting(X_revalidation[i], y_revalidation[i] + 1))
X_rerevalidation = torch.cat(X_rerevalidation, 0)
y_rerevalidation = y_validation[y_validation != 10].to(torch.long)

dim_size = [512, 128, 5]
one_model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    padding=0),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(832, dim_size[1]),
    torch.nn.BatchNorm1d(dim_size[1]),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_size[1], dim_size[2]),
    )

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(one_model.parameters())
batch_size = 5
retrain_data = TensorDataset(X_retrain, y_retrain)
revalidation_data = TensorDataset(X_revalidation, y_revalidation)
retrain_loading = torch.utils.data.DataLoader(retrain_data, batch_size=batch_size)
revalidation_loading = torch.utils.data.DataLoader(revalidation_data, batch_size=batch_size)
train(one_model, retrain_loading, revalidation_loading)
ind = np.random.choice(len(X_test_processed))
predict_nums(X_test_processed[ind])
one_model.eval()
splitted_digits = splitting(X_test_processed[ind], predict_nums(X_test_processed[ind]))

dim_size = [256, 256, 128, 128, 10]
second_model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(128),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(256),
    torch.nn.ReLU(),
    torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(256),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    padding=0),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(256, dim_size[1]),
    torch.nn.BatchNorm1d(256),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_size[1], dim_size[2]),
    torch.nn.BatchNorm1d(dim_size[2]),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_size[2], dim_size[3]),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(second_model.parameters())
batch_size = 40
reretrain = TensorDataset(X_reretrain, y_reretrain)
rerevalidation = TensorDataset(X_rerevalidation, y_rerevalidation)
reretrain_loading = torch.utils.data.DataLoader(reretrain, batch_size=batch_size)
rerevalidation_loading = torch.utils.data.DataLoader(rerevalidation, batch_size=batch_size)
train(second_model, reretrain_loading, rerevalidation_loading, iterations=500000, update=500)
train_acc = accuracy(predict(retrain_loading, write_to_csv=False), y_train)
val_acc = accuracy(predict(revalidation_loading, write_to_csv=False), y_validation)
print(f'accuracy: training {train_acc}, validation {val_acc}')
predict(X_test_processed)