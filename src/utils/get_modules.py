import torch
from .namers import classifier_ckpt_namer


def create_classifier(args):

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.neural_net.architecture == "resnetwide":
        from ..models.resnet import ResNetWide_constant_fraction
        classifier = ResNetWide_constant_fraction(
            num_outputs=10).to(device)

    elif args.neural_net.architecture == "resnet":
        from ..models.resnet import ResNet_constant_fraction
        classifier = ResNet_constant_fraction(
            num_outputs=10).to(device)

    elif args.neural_net.architecture == "vgg":
        from ..models.vgg import VGG_constant_fraction
        classifier = VGG_constant_fraction().to(device)

    else:
        raise NotImplementedError

    return classifier


def load_classifier(args):

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    classifier = create_classifier(args)

    try:
        param_dict = torch.load(classifier_ckpt_namer(args),
                                map_location=torch.device(device),)
    except:
        raise FileNotFoundError(classifier_ckpt_namer(args))

    if "module" in list(param_dict.keys())[0]:
        for _ in range(len(param_dict)):
            key, val = param_dict.popitem(False)
            param_dict[key.replace("module.", "")] = val

    classifier.load_state_dict(param_dict)

    print(f"Classifier: {classifier_ckpt_namer(args)}")

    return classifier
