import torch
from torchvision import datasets, transforms
from os.path import join


def cifar10(args):

    use_cuda = args.use_gpu and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(
        root=join(args.directory, 'data', 'original_datasets'),
        train=True,
        download=True,
        transform=transform_train,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.neural_net.train_batch_size, shuffle=True, num_workers=2
    )

    testset = datasets.CIFAR10(
        root=join(args.directory, 'data', 'original_datasets'),
        train=False,
        download=True,
        transform=transform_test,
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.neural_net.test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader
