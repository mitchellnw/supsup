import sys, os

sys.path.append(os.path.abspath("."))
from main import main as run
from args import args


def main():
    args.set = 'MNISTPerm'
    args.multigpu = [0]
    args.model = 'LeNet'
    args.conv_type = 'VectorizedBatchEnsembles'
    args.conv_init = 'kaiming_normal'
    args.name = 'id=batche_lenet_mnistperm'
    args.epochs = 3

    args.adapt_lrs = [0]
    args.eval_ckpts = [10, 50, 100, 150, 200]
    args.num_tasks = 250
    args.adaptor = "gt"
    args.output_size = 10

    args.lr = 1e-2
    args.train_weight_lr = 1e-4
    args.optimizer = 'adam'
    args.no_scheduler = True
    args.iter_lim = 1000
    args.train_weight_tasks = 1
    args.save = True

    args.data = '~/data'
    args.log_dir = "~/checkpoints/test"

    run()


if __name__ == "__main__":
    main()
