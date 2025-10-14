"""Main entry point for SAFT (Semantic-aware Adversarial Fine-Tuning) training pipeline"""

from __future__ import print_function
import torch
from config import parse_option
from training import TrainingManager


def main(args):
    """Main entry point for training"""
    global device
    device = torch.device(f"cuda:{args.gpu}")

    trainer = TrainingManager(args)
    trainer.run_training()


if __name__ == "__main__":
    """Main entry point - parse args and run training"""
    args = parse_option()
    main(args)
