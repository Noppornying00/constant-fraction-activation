from tqdm import tqdm
from deepillusion.torchdefenses import adversarial_test
from .utils.read_datasets import cifar10
import torch
from .utils.get_modules import load_classifier
from foolbox import PyTorchModel, accuracy
from foolbox.attacks import LinfPGD, L2PGD
from torchattacks import PGDL2, Square, PGD
from art.attacks.evasion import SquareAttack
from art.estimators.classification import PyTorchClassifier
from .utils.device import determine_device
import os
from .parameters import get_arguments


def main():

    args = get_arguments()

    device = determine_device(args)

    model = load_classifier(args)

    print(args)

    model = model.to(device)
    model.eval()

    breakpoint()
    for p in model.parameters():
        p.requires_grad = False

    _, test_loader = cifar10(args)
    test_loss, test_acc = adversarial_test(model, test_loader)
    print(f"Clean \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

    foolbox_model = PyTorchModel(model, bounds=(0, 1))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    art_model = PyTorchClassifier(
        model=model,
        clip_values=(0., 1.),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    clean_total = 0
    adv_total = 0

    for batch_idx, items in enumerate(
        pbar := tqdm(test_loader, desc="Attack progress", leave=False)
    ):
        if args.attack.nb_imgs > 0 and args.attack.nb_imgs < (batch_idx + 1) * args.neural_net.test_batch_size:
            break

        data, target = items
        data = data.to(device)
        target = target.to(device)

        clean_acc = accuracy(foolbox_model, data, target)
        clean_total += int(clean_acc*args.neural_net.test_batch_size)

        if args.attack.library == "foolbox":
            if args.attack.method == "PGD":
                if args.attack.norm == "inf":
                    attack = LinfPGD(
                        abs_stepsize=args.attack.step_size, steps=args.attack.nb_steps)

                if args.attack.norm == 2:
                    attack = L2PGD(abs_stepsize=args.attack.step_size,
                                   steps=args.attack.nb_steps)

            else:
                raise NotImplementedError

        elif args.attack.library == "art":
            if args.attack.method == "square":
                if args.attack.norm == "inf":
                    data = data.cpu().numpy()
                    target = target.cpu().numpy()
                    attack = SquareAttack(estimator=art_model, norm="inf", max_iter=10000, eps=args.attack.budget,
                                          p_init=0.8, nb_restarts=1, batch_size=args.neural_net.test_batch_size)

            else:
                raise NotImplementedError

        elif args.attack.library == "torchattacks":
            if args.attack.method == "PGD":
                if args.attack.norm == "inf":
                    attack = PGD(model, eps=args.attack.budget, alpha=args.attack.step_size,
                                 steps=args.attack.nb_steps, random_start=False)

                if args.attack.norm == 2:
                    attack = PGDL2(model, eps=args.attack.budget, alpha=args.attack.step_size,
                                   steps=args.attack.nb_steps, random_start=False, eps_for_division=1e-10)

            elif args.attack.method == "square":
                if args.attack.norm == "inf":

                    attack = Square(model, norm='Linf', eps=args.attack.budget,
                                    n_queries=10000, n_restarts=1, p_init=.8)

                if args.attack.norm == 2:
                    attack = Square(model, norm='L2', eps=args.attack.budget,
                                    n_queries=10000, n_restarts=1, p_init=.8)

        if args.attack.library == "foolbox":

            raw_advs, clipped_advs, success = attack(
                foolbox_model, data, target, epsilons=args.attack.budget)

        elif args.attack.library == "art":

            clipped_advs = attack.generate(x=data)
            clipped_advs = torch.from_numpy(clipped_advs).to(device)
            target = torch.from_numpy(target).to(device)

        elif args.attack.library == "torchattacks":
            clipped_advs = attack(data, target)

        with torch.no_grad():
            attack_out = model(clipped_advs)
            pred_attack = attack_out.argmax(dim=1, keepdim=True)

            attack_correct = pred_attack.eq(
                target.view_as(pred_attack)).sum().item()

            adv_total += int(attack_correct)
            adv_acc_sofar = adv_total / \
                ((batch_idx+1) * args.neural_net.test_batch_size)

        pbar.set_postfix(
            Adv_ac=f"{adv_acc_sofar:.4f}", refresh=True,
        )

    print(f"clean accuracy:  {clean_total / 10000 * 100:.2f} %")
    print(f"adv accuracy:  {adv_total / 10000 * 100:.2f} %")

    if args.attack.savefig:

        import matplotlib.pyplot as plt
        import matplotlib
        from .utils import plot_settings

        clipped_advs = clipped_advs[~attack_correct.squeeze()]
        data = data[~attack_correct.squeeze()]
        perturbations = clipped_advs-data

        matplotlib.rc('text', usetex=False)

        fig, axes = plt.subplots(nrows=8, ncols=3, figsize=(11, 5))

        for i, ax in enumerate(axes.flat):
            ax.set_axis_off()
            if i % 3 == 0:
                im = ax.imshow(data.detach().cpu()[
                    i//3].numpy().transpose(1, 2, 0))

            elif i % 3 == 1:
                im = ax.imshow(clipped_advs.detach().cpu()[
                    i//3].numpy().transpose(1, 2, 0))

            elif i % 3 == 2:
                im = ax.imshow((perturbations.detach().cpu()[
                    i//3].numpy().transpose(1, 2, 0)/args.attack.budget*0.5+0.5).clip(0.0, 1.0))
                # breakpoint()

        plt.tight_layout()

        if not os.path.exists('figs'):
            os.makedirs('figs')

        plt.savefig('figs/attacks.pdf')


if __name__ == "__main__":
    main()
