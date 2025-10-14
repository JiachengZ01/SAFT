"""Training logic for SAFT training pipeline"""

import os
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import clip
from models.model import multiGPU_CLIP_classwise, clip_img_preprocessing
from attacks.pgd_attack import pgd_CLIP
from helper import (
    AverageMeter, init_wandb, load_train_dataset, load_val_datasets,
    get_text_prompts_train, get_text_prompts_val, save_checkpoint,
    cosine_lr, criterion, attention_map_text
)
from model_setup import ModelSetup
from validation import validate
import wandb


class TrainingManager:
    """Manages the training process including setup and execution"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}")
        self._setup_environment()

    def _setup_environment(self):
        """Setup device, normalize parameters, and set random seed"""
        # Normalize epsilon values to [0,1] range
        self.args.train_eps /= 255.0
        self.args.test_eps /= 255.0
        self.args.train_stepsize /= 255.0
        self.args.test_stepsize /= 255.0

        # Set random seed for reproducibility
        if self.args.seed is not None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            print(f"seed is set as: {self.args.seed}")
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        if self.args.wandb and not self.args.checkpoint:
            if self.args.ncaps == 0:
                init_wandb(
                    project_name="SAFT",
                    model_name=f'TGA-ZSR-seed{self.args.seed}-BS{self.args.batch_size}-{self.args.arch}',
                    config=vars(self.args),
                )
            else:
                init_wandb(
                    project_name="SAFT",
                    model_name=f'{self.args.Method}-{self.args.ncaps}caps-seed{self.args.seed}-BS{self.args.batch_size}-{self.args.arch}',
                    config=vars(self.args),
                )
        else:
            wandb.init(mode='disabled')

    def run_training(self):
        """Main training pipeline"""
        self._init_wandb()
        self._print_args()

        # Initialize models
        model_setup = ModelSetup(self.args, self.device)
        model, frozen_model = model_setup.create_models()
        optimizer = model_setup.setup_optimizer(model)
        model_setup.load_checkpoint(model)

        # Setup data and training components
        train_loader, val_loader_list, val_dataset_name = self._setup_data()
        texts_train, texts_list = self._setup_text_prompts(model, train_loader.dataset, val_loader_list, val_dataset_name)

        # Training setup
        scaler = GradScaler()
        total_steps = len(train_loader) * self.args.epochs
        scheduler = cosine_lr(optimizer, self.args.learning_rate, self.args.warmup, total_steps)
        cudnn.benchmark = True

        # Create model directory
        self.args.model_folder = os.path.join(self.args.model_dir, self.args.filename)
        os.makedirs(self.args.model_folder, exist_ok=True)

        if not self.args.checkpoint:
            # Training phase
            for epoch in range(self.args.epochs):
                train(train_loader, texts_train, model, frozen_model,
                      optimizer, scheduler, scaler, epoch, self.device, self.args)

                # Save checkpoint after each epoch (will be overwritten by next epoch)
                save_checkpoint({
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                    "vision_encoder_state_dict": model.module.visual.state_dict(),
                }, self.args, is_final=False)

            # Save final model after all epochs
            save_checkpoint({
                "epoch": self.args.epochs,
                "optimizer": optimizer.state_dict(),
                "vision_encoder_state_dict": model.module.visual.state_dict(),
            }, self.args, is_final=True)

            # Final validation
            adv_acc, clean_acc = validate(val_loader_list, val_dataset_name, texts_list, model, frozen_model, self.device, self.args, self.args.epochs, texts_train)
        else:
            # Evaluation only
            adv_acc, clean_acc = self._run_evaluation(model, frozen_model, val_loader_list, val_dataset_name, texts_list, texts_train)

        # Log final results
        print(f"Adversarial accuracy: {adv_acc}")
        print(f"Clean accuracy: {clean_acc}")

        if self.args.wandb:
            wandb.run.summary["last_adv_acc"] = adv_acc
            wandb.run.summary["last_clean_acc"] = clean_acc
            wandb.finish()

    def _print_args(self):
        """Print configuration arguments"""
        print(f"Arguments:\n{'-' * 20}")
        for arg, value in vars(self.args).items():
            print(f"{arg}: {value}")
        print(f"{'-' * 20}")

    def _setup_data(self):
        """Setup training and validation data loaders"""
        # Load datasets
        train_dataset = load_train_dataset(self.args)
        val_dataset_name = self._get_val_dataset_names()
        val_dataset_list = load_val_datasets(self.args, val_dataset_name)

        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            pin_memory=True, num_workers=4,
        )

        val_loader_list = [
            DataLoader(dataset, batch_size=self.args.batch_size, pin_memory=True,
                      shuffle=False, num_workers=4)
            for dataset in val_dataset_list
        ]

        return train_loader, val_loader_list, val_dataset_name

    def _get_val_dataset_names(self):
        """Get validation dataset names based on configuration"""
        if self.args.testdata is not None:
            return self.args.testdata

        # base_datasets = ['tinyImageNet','cifar10', 'cifar100','STL10','Food101','oxfordpet',
        #                 'flowers102','dtd','EuroSAT','fgvc_aircraft','Caltech101','Caltech256',
        #                 'StanfordCars','PCAM']

        base_datasets = ['tinyImageNet', 'cifar10', 'cifar100','STL10']

        return base_datasets

    def _setup_text_prompts(self, model, train_dataset, val_dataset_list, val_dataset_name):
        """Setup text prompts for training and validation"""
        template = "This is a photo of a {}"
        print(f"template: {template}")

        texts_train = get_text_prompts_train(model, self.args, train_dataset, self.device, template=template)
        # If a list of DataLoaders was passed in, convert to their underlying Datasets
        datasets_for_val = [dl.dataset for dl in val_dataset_list]
        texts_list = get_text_prompts_val(datasets_for_val, val_dataset_name, template=template)

        return texts_train, texts_list

    def _run_evaluation(self, model, frozen_model,
                       val_loader_list, val_dataset_name, texts_list, texts_train):
        """Run evaluation on pre-trained model"""
        # Load baseline model
        model_setup = ModelSetup(self.args, self.device)
        model_setup.load_baseline_model(model)

        # Setup text prompts - extract datasets from loaders
        datasets_for_val = [dl.dataset for dl in val_loader_list]
        texts_list = get_text_prompts_val(datasets_for_val, val_dataset_name, template=self.args.template)
        print(f"template for evaluation: {self.args.template}")

        return validate(val_loader_list, val_dataset_name, texts_list, model, frozen_model,
                          self.device, self.args, self.args.epochs, None)


def train(train_loader, texts, model, frozen_model,
          optimizer, scheduler, scaler, epoch, device, args):
    """Train model for one epoch with adversarial training"""
    # Initialize metrics tracking
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    # Set models to training mode
    model.module.visual.train()

    # Training parameters
    num_batches_per_epoch = len(train_loader)
    alpha = args.train_stepsize
    attack_iters = args.train_numsteps
    end = time.time()

    # Prepare text tokens for training
    text_tokens = clip.tokenize(texts).to(device)
    text_tokens = text_tokens.view(args.ncaps, 200, -1)

    for i, (images, target) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)

        # Update learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad()

        images, target = images.to(device), target.to(device)

        # Generate adversarial examples
        delta = pgd_CLIP(
            model, images, target, text_tokens,
            alpha, attack_iters, "l_inf", device, ncaps=args.ncaps,
            epsilon=args.train_eps,
        )

        with torch.autocast(device_type='cuda'):
            # Enable gradient computation
            for param in model.parameters():
                param.requires_grad = True

            # Forward pass
            adv_images = clip_img_preprocessing(images + delta, device)
            clean_images = clip_img_preprocessing(images, device)

            logits_per_image, _, text_features = multiGPU_CLIP_classwise(
                model, adv_images, text_tokens
            )
            text_features = text_features[:, target, :]

            # Calculate attention maps for loss computation
            attack_attention = attention_map_text(
                text_features, model, adv_images, args
            ).view(adv_images.size()[0], -1)

            clean_orig_attention = attention_map_text(
                text_features, frozen_model, clean_images, args
            ).view(clean_images.size()[0], -1)

            clean_target_attention = attention_map_text(
                text_features, model, clean_images, args
            ).view(clean_images.size()[0], -1)

            # Compute multi-component loss
            loss_TeCoA, loss_AM1, loss_AM2 = criterion(
                logits_per_image, target, attack_attention,
                clean_orig_attention, clean_target_attention, device, args,
            )

            total_loss = loss_TeCoA + loss_AM1 + loss_AM2

            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Clamp logit scale as in original CLIP
        model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        # Update metrics
        losses.update(total_loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging and debug break
        if i % args.print_freq == 0:
            if args.wandb:
                wandb.log({
                    "train/batch_time": batch_time.avg,
                    "train/total_train_loss": losses.avg,
                    "train/TeCoA_loss": loss_TeCoA.item(),
                    "train/AM1_loss": loss_AM1.item(),
                    "train/AM2_loss": loss_AM2.item(),
                    "train_step": step,
                })
            if args.debug:
                break

    return losses.avg
