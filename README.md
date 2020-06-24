# Supermasks in Superposition for Continual Learning

Code for running experiments is located in the `/experiments` folder and grouped by GG, GNu, and NNs in accordance with our paper's hierarchy. 

## Environment set-up

We include requirements file in `requirements.txt`. Make a new virtual environment in your favorite environment manager (conda, virtualenv) and run `pip install -r requirements.txt`.

## GG Experiments

### Directory Structure

The `experiments/GG/splitcifar100/` folder contains the following experiment scripts:

1. `experiments/GG/splitcifar100/rn18-batche-randw.py` -> Corresponds to _BatchE (GG) - Rand W_ in Figure 3 (right)
2. `experiments/GG/splitcifar100/rn18-separate-heads.py` -> Corresponds to _Separate Heads_ in Figure 3 (right)
3. `experiments/GG/splitcifar100/rn18-separate-heads-randw.py` -> Corresponds to _Separate Heads - Rand W_ in Figure 3 (right)
4. `experiments/GG/splitcifar100/rn18-supsup.py` -> Corresponds to _SupSup_ (our method) in Figure 3 (right)
5. `experiments/GG/splitcifar100/rn18-supsup-transfer.py` -> Corresponds to _SupSup Transfer_ (our method with transfer) in Figure 3 (right)
6. `experiments/GG/splitcifar100/rn18-upperbound.py` -> Corresponds to _Upper Bound_ in Figure 3 (right)


The `splitimagenet/` folder contains one experiment script:

`experiments/GG/splitimagenet/rn50-supsup.py` -> Corresponds to all 3 runs of _Sup Sup_ in Figure 3 (left)

The actual settings for these experiments (e.g. hyperparameters) are stored in `experiments/GG/splitcifar100/configs` and `experiments/GG/splitimagenet/configs`

### How to run an experiment

Go to the root directory of this code repository and invoke one of the scripts from above with `--gpu-sets`, `--seeds`, and `--data` flags, e.g.

```
python experiments/GG/splitcifar100/rn18-supsup.py --gpu-sets="0|1|2|3" --data=/path/to/dataset/parent --seeds 1
```

The `--data` flag is the path to the folder which contains the required dataset, in this case CIFAR100 or ImageNet, which we then split into tasks. CIFAR100 will be automatically downloaded if it's not in `--data`, ImageNet will not. `--seeds` says how many seeds (from 0 to `--seeds - 1` to evaluate on. For all of our reported SplitCIFAR100 experiments we use 5. Our reported experiments for SplitImageNet are with 1 seed (with fixed ImageNet split). The default number of seeds in this repo is 1. 

Since we are in the GG scenario, these models can be trained on each task individually. As such these scripts are built to take advantage of parallelism. The `--gpu-sets` command takes comma-separated sets of GPUs separated by `|`. For example, `--gpu-sets="0|1|2|3"` means that each experiment will be run individually on a GPU with ID in [0, 1, 2, 3]. If you want to use multiple GPUs per experiment, say for ResNet-50 on SplitImagenet, you can specify this by using comma-separated lists. For example, `--gpu-sets="0,1|2,3"` means that each task will be trained invidually (in parallel) either on GPUs {0, 1} or {2, 3}. Specifying a lone gpu, `--gpu-sets=0`, means that experiments will be run sequentially on GPU 0.

### Where are the results stored?

Results are automatically stored after each run in the `runs/<experiment-name>` folder, where `<experiment-name>` is the name of the script file (sans `.py` extension). The actual numbers corresponding to our plot are stored in `runs/<experiment-name>/results.csv` where each row has a self-explanatory `Name` column describing what the result is.


## GNu/NNs Experiments

The `experiments/GNu/MNISTPerm` contains the MNISTPerm experiments.

E.g. `experiments/GNu/MNISTPerm/LeNet-250-tasks` and `experiments/GNu/MNISTPerm/FC-250-tasks` correspond to Figure 4 (left) and (right) respectively and `experiments/GNu/MNISTPerm/LeNet-2500-tasks` correspond to the GNu experiments in Figure 5.

The `experiments/GNu/MNISTRotate` contains the MNISTRotate experiments in Figure 6, and the `experiments/GNu/SplitMNIST` contains the HopSupSup experiment.

The `experiments/NNs` contains the NNs experiments, which appear on Figure 5.

For example, an experiment can be run with
```bash
python experiments/GNu/MNISTPerm/LeNet-2500-tasks/supsup_h.py
```
where `args.data` should point to a directory containing the dataset and checkpoints/results will be logged at `args.log_dir`.
These can be changed in the python file. The ablations can be reproduced by e.g. changing output size.