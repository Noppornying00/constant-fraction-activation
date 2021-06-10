"""
Hyper-parameters
"""

import argparse
from os import environ, path
import toml
import json
from types import SimpleNamespace


def make_config_obj(args):

    config = toml.load(path.join(args.directory, "src", "config.toml"))
    config = json.loads(json.dumps(config),
                        object_hook=lambda d: SimpleNamespace(**d))

    for attr in "start_fraction", "end_fraction", "directory":
        setattr(config, attr, getattr(args, attr))

    for arg, val in args.__dict__.items():
        for subconfig in ["neural_net", "optimizer", "attack"]:
            if subconfig in arg:
                setattr(getattr(config, subconfig),
                        arg.replace(subconfig+"_", ""), val)

    if config.attack.norm != 'inf':
        config.attack.norm = int(config.attack.norm)

    return config


def get_arguments():
    """ Hyperparameters and other configuration items"""

    if environ.get("PROJECT_PATH") is not None:
        directory = environ["PROJECT_PATH"]
    else:
        import pathlib

        directory = path.dirname(path.abspath(__file__))
        if "src" in directory:
            directory = directory.replace("src", "")

    if directory[-1] == "/" and directory[-2] == "/":
        directory = directory[:-1]
    elif directory[-1] != "/":
        directory += "/"

    parser = argparse.ArgumentParser(
        description="")

    parser.add_argument(
        "--directory",
        type=str,
        default=directory,
        metavar="",
        help="Directory of experiments",
    )

    parser.add_argument(
        "--start_fraction",
        type=float,
        default=-1.,
    )

    parser.add_argument(
        "--end_fraction",
        type=float,
        default=-1.,
    )

    # Adversarial testing parameters
    neural_net = parser.add_argument_group(
        "neural_net", "Neural-net related config"
    )
    neural_net.add_argument(
        "--neural_net_epochs",
        type=int,
        default=70,
        metavar="",
        help="Number of epochs in training",
    )

    neural_net.add_argument(
        "--optimizer_lr",
        type=float,
        default=0.001,
        help="learning rate for training",
    )

    # Adversarial testing parameters
    attack = parser.add_argument_group(
        "attack", "Adversarial testing related config"
    )

    attack.add_argument(
        "--attack_transfer_file",
        type=str,
        default=None,
        help="Source file for the transfer attack (only filename)",
    )

    attack.add_argument(
        "-at_lib",
        "--attack_library",
        type=str,
        default="foolbox",
        choices=["foolbox", "art", "torchattacks"],
        help="Attack library",
    )

    attack.add_argument(
        "-at_method",
        "--attack_method",
        type=str,
        default="PGD",
        choices=[
            "PGD",
            "square"
        ],
        help="Attack method for white/semiwhite box attacks",
    )

    attack.add_argument(
        "-at_norm",
        "--attack_norm",
        type=str,
        default="inf",
        metavar="inf/p",
        help="Which attack norm to use",
    )
    attack.add_argument(
        "-at_eps",
        "--attack_budget",
        type=float,
        default=(8.0 / 255.0),
        metavar="",
        help="attack budget",
    )
    attack.add_argument(
        "-at_ss",
        "--attack_step_size",
        type=float,
        default=(1.0 / 255.0),
        metavar="",
        help="Step size for PGD",
    )
    attack.add_argument(
        "-at_ni",
        "--attack_nb_steps",
        type=int,
        default=40,
        metavar="",
        help="Number of steps for PGD",
    )

    attack.add_argument(
        "-at_rand",
        "--attack_rand",
        type=bool,
        default=True,
        help="randomly initialize PGD attack",
    )
    attack.add_argument(
        "-at_nr",
        "--attack_nb_restarts",
        type=int,
        default=100,
        metavar="",
        help="number of restarts for PGD",
    )

    args = parser.parse_args()

    config = make_config_obj(args)

    return config
