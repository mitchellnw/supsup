import sys, os

sys.path.append(os.path.abspath("."))
from main import main as run
from args import args


def main():
    args.data_to_repeat = 1
    args.set = 'MNISTPerm'
    args.multigpu = [0]
    args.model = 'LeNet'
    args.conv_type = 'FastMultitaskMaskConv'
    args.conv_init = 'signed_constant'

    args.eval_ckpts = [10, 500, 1000, 1500, 2000, 2500]
    args.num_tasks = 2500
    args.adaptor = "se_oneshot_entropy_minimization"
    args.output_size = 500

    args.name = f'id=supsup_h_mnistperm_long'

    args.lr = 1e-4
    args.optimizer = 'rmsprop'
    args.iter_lim = 1000
    args.epochs = 3

    args.save = True
    args.no_scheduler = True
    args.unshared_labels = True
    args.data_to_repeat = 1

    args.data = '~/data'
    args.log_dir = "~/checkpoints/test"

    run()


if __name__ == "__main__":
    main()
