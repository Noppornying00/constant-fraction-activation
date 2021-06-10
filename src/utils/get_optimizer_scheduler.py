import torch.optim as optim


def get_optimizer_scheduler(args, model):

    if args.optimizer.name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.optimizer.lr,
            momentum=args.optimizer.momentum,
            weight_decay=args.optimizer.weight_decay,
        )
    elif args.optimizer.name == "rms":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.optimizer.lr,
            weight_decay=args.optimizer.weight_decay,
            momentum=args.optimizer.momentum,
        )

    elif args.optimizer.name == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay
        )

    else:
        raise NotImplementedError

    if args.optimizer.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[35], gamma=0.1
        )

    elif args.optimizer.lr_scheduler == "mult":

        def lr_fun(epoch):
            if epoch % 3 == 0:
                return 0.962
            else:
                return 1.0

        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fun)
    else:
        raise NotImplementedError

    return optimizer, scheduler
