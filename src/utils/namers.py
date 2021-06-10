from os.path import join


def classifier_params_string(args):
    classifier_params_string = args.neural_net.architecture

    classifier_params_string += f"_{args.optimizer.name}"

    classifier_params_string += f"_{args.optimizer.lr_scheduler}"

    classifier_params_string += f"_{args.optimizer.lr:.4f}"

    classifier_params_string += f"_ep_{args.neural_net.epochs}"

    if args.start_fraction > 0.0:
        classifier_params_string += f"_sf_{args.start_fraction:.3f}"
        classifier_params_string += f"_ef_{args.end_fraction:.3f}"

    return classifier_params_string


def classifier_ckpt_namer(args):

    file_path = join(args.directory, 'checkpoints',
                     'classifiers')

    file_path = join(file_path, classifier_params_string(args) + '.pt')

    return file_path


def classifier_log_namer(args):

    file_path = join(args.directory, 'logs')

    file_path = join(file_path, classifier_params_string(args) + '.log')

    return file_path
