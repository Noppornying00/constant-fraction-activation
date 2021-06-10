import torch
import torch.nn as nn
from tqdm import tqdm


def train(model, train_loader, optimizer, scheduler=None):
    """ Train given model with train_loader and optimizer """

    model.train()
    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target, in train_loader:
        if isinstance(data, list):
            data = data[0]
            target = target[0]

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()

    scheduler.step()

    train_size = len(train_loader.dataset)

    return train_loss / train_size, train_correct / train_size


def train_constant_fraction(model, train_loader, optimizer, scheduler, fraction):

    model.train()
    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target, in train_loader:
        if isinstance(data, list):
            data = data[0]
            target = target[0]

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()

        model.adjust_bias(fraction)

        # print(model.get_activation_fractions())

    scheduler.step()

    train_size = len(train_loader.dataset)

    return train_loss / train_size, train_correct / train_size


def test(model, test_loader):

    model.eval()

    device = model.parameters().__next__().device

    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if isinstance(data, list):
                data = data[0]
                target = target[0]

            data, target = data.to(device), target.to(device)

            output = model(data)
            cross_ent = nn.CrossEntropyLoss()
            test_loss += cross_ent(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

            # print(model.get_activation_fractions())

    test_size = len(test_loader.dataset)

    return test_loss / test_size, test_correct / test_size
