"""Configuration management for SAFT training pipeline"""

import argparse
from helper import str2bool


class Config:
    """Configuration manager for training parameters"""

    def __init__(self):
        self.parser = argparse.ArgumentParser("Adapting CLIP for zero-shot adv robustness")
        self._setup_args()

    def _setup_args(self):
        # Training parameters
        self.parser.add_argument("--print_freq", type=int, default=50, help="print frequency")
        self.parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
        self.parser.add_argument("--validate_freq", type=int, default=10, help="validate frequency")
        self.parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
        self.parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
        self.parser.add_argument("--num_workers", type=int, default=8)

        # Optimization parameters
        self.parser.add_argument("--Method", type=str, default="TGA-ZSR")
        self.parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
        self.parser.add_argument("--warmup", type=int, default=1000, help="number of steps to warmup for")
        self.parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

        # Adversarial attack parameters
        self.parser.add_argument("--train_eps", type=float, default=1, help="training epsilon")
        self.parser.add_argument("--train_numsteps", type=int, default=2)
        self.parser.add_argument("--train_stepsize", type=int, default=1)
        self.parser.add_argument("--test_eps", type=float, default=1, help="test epsilon")
        self.parser.add_argument("--test_numsteps", type=int, default=100)
        self.parser.add_argument("--test_stepsize", type=int, default=1)

        # Model parameters
        self.parser.add_argument("--patience", type=int, default=1000)
        self.parser.add_argument("--model", type=str, default="clip")
        self.parser.add_argument("--arch", type=str, default="vit_b32", choices=["vit_b32", "vit_l14"])
        self.parser.add_argument("--method", type=str, default="null_patch",
                                choices=["padding", "random_patch", "fixed_patch", "null_patch"],
                                help="choose visual prompting method")

        # Dataset parameters
        self.parser.add_argument("--root", type=str, default="./data", help="dataset")
        self.parser.add_argument("--dataset", type=str, default="tinyImageNet",
                                choices=["cifar100", "ImageNet", "cifar10", "tinyImageNet"],
                                help="Data set for training")
        self.parser.add_argument("--image_size", type=int, default=224, help="image size")
        self.parser.add_argument("--text", type=str, default="normal",
                                choices=["normal", "semantic_ensemble"],
                                help="text prompt for training")

        # Other parameters
        self.parser.add_argument("--seed", type=int, default=None, help="seed for initializing training")
        self.parser.add_argument("--index", type=int, default=1, help="index for experiments")
        self.parser.add_argument("--model_dir", type=str, default="./save/models", help="path to save models")
        self.parser.add_argument("--filename", type=str, default=None, help="filename to save")
        self.parser.add_argument("--trial", type=int, default=1, help="number of trials")
        self.parser.add_argument("--resume", type=str, default=None, help="path to resume from checkpoint")
        self.parser.add_argument("--gpu", type=int, default=0, help="gpu to use")
        self.parser.add_argument("--debug", action="store_true")
        self.parser.add_argument("--VPbaseline", action="store_true")
        self.parser.add_argument("--attack", choices=["pgd", "autoattack", "cw"], default="pgd")
        self.parser.add_argument("--noimginprop", action="store_true")
        self.parser.add_argument("--ncaps", type=int, default=1, help="number of captions per image")
        self.parser.add_argument("--checkpoint", action="store_true")
        self.parser.add_argument("--template", type=str, default="This is a photo of a {}")
        self.parser.add_argument("--baseline", type=str,
                                choices=["PMG-AFT", "TeCoA", "FARE", "TGA-ZSR", "ours", "CLIP"],
                                default="ours")
        self.parser.add_argument("--random-target", action="store_true")
        self.parser.add_argument("--autoattack", action="store_true")
        self.parser.add_argument("--test-continue", action="store_true")
        self.parser.add_argument("--MLLM", action="store_true")

        # Fine-tuning parameters
        self.parser.add_argument("--last_num_ft", type=int, default=0)
        self.parser.add_argument("--adaptation_method", type=str, default="FT",
                                choices=["VPT", "FT"], help="choose visual adaptation method")
        self.parser.add_argument("--Distance_metric", type=str, default="l2",
                                choices=["cos", "l2", "l1"],
                                help="Select the distance measure in the loss function")
        self.parser.add_argument("--atten_methods", type=str, default="text", choices=["text", "visual"])
        self.parser.add_argument("--Alpha", type=float, default=0.08, help="L_AR in Equ.6")
        self.parser.add_argument("--Beta", type=float, default=0.05, help="L_AMC in Equ.7")
        self.parser.add_argument("--testdata", type=str, nargs="+")
        self.parser.add_argument("--wandb", type=str2bool, default=True, help="Use Weights & Biases for logging")

    def parse_args(self):
        """Parse arguments and generate filename"""
        args = self.parser.parse_args()
        self._generate_filename(args)
        return args

    def _generate_filename(self, args):
        """Generate filename based on configuration parameters"""
        args.filename = "{}_{}_{}_{}_lr-{}_decay-{}_bsz-{}_warmup-{}_trial-{}_Alpha-{}_Beta-{}_distance-{}_atten_methods-{}_ncaps-{}_seed-{}_MLLM-{}".format(
            args.Method, args.dataset, args.model, args.arch, args.learning_rate,
            args.weight_decay, args.batch_size, args.warmup, args.trial,
            args.Alpha, args.Beta, args.Distance_metric, args.atten_methods,
            args.ncaps, args.seed, args.MLLM
        )


def parse_option():
    """Create and return parsed configuration"""
    config = Config()
    return config.parse_args()
