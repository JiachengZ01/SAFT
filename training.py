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
                    project_name="CLIP_robust_finetune_text_enhanced",
                    model_name=f'TGA-ZSR-seed{self.args.seed}-BS{self.args.batch_size}-{self.args.arch}',
                    config=vars(self.args),
                )
            else:
                init_wandb(
                    project_name="CLIP_robust_finetune_text_enhanced",
                    model_name=f'{self.args.Method}-{self.args.ncaps}caps-index{self.args.index}-BS{self.args.batch_size}-{self.args.arch}',
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
        prompter, add_prompter, optimizer = model_setup.setup_prompters_and_optimizer(model)
        best_acc_adv, best_acc_clean = model_setup.load_checkpoint(model, prompter, add_prompter)

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
                train(train_loader, texts_train, model, frozen_model, prompter,
                      add_prompter, optimizer, scheduler, scaler, epoch, self.device, self.args)

            # Save final model
            save_checkpoint({
                "epoch": self.args.start_epoch + self.args.epochs,
                "optimizer": optimizer.state_dict(),
                "vision_encoder_state_dict": model.module.visual.state_dict(),
            }, self.args, is_best=True)

            # Final validation
            adv_acc_mean, clean_acc_mean = validate(val_loader_list, val_dataset_name, texts_list,
                                                   model, frozen_model, self.device, prompter,
                                                   add_prompter, self.args, self.args.epochs, texts_train)
        else:
            # Evaluation only
            adv_acc_mean, clean_acc_mean = self._run_evaluation(model, frozen_model, prompter,
                                                              add_prompter, val_loader_list,
                                                              val_dataset_name, texts_list, texts_train)

        # Log final results
        print(f"Averaged adversarial accuracy: {adv_acc_mean}")
        print(f"Averaged clean accuracy: {clean_acc_mean}")

        if self.args.wandb:
            wandb.run.summary["last_avg_adv_acc"] = adv_acc_mean
            wandb.run.summary["last_avg_clean_acc"] = clean_acc_mean
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

        # Create data loaders
        def collate_fn(batch):
            images, target, caption = zip(*batch)
            images = torch.stack(images, 0)
            target = torch.tensor(target)
            text_tokens = clip.tokenize(caption, truncate=True)
            return images, target, text_tokens

        generator = torch.Generator()
        if self.args.seed is not None:
            generator.manual_seed(self.args.seed)

        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            pin_memory=True, collate_fn=collate_fn, num_workers=4,
            generator=generator if self.args.seed is not None else None,
        )

        val_loader_list = [
            DataLoader(dataset, batch_size=self.args.batch_size * 2, pin_memory=True,
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

        base_datasets = ['tinyImageNet','cifar10', 'cifar100','STL10']

        return base_datasets

    def _setup_text_prompts(self, model, train_dataset, val_dataset_list, val_dataset_name):
        """Setup text prompts for training and validation"""
        template = "This is a photo of a {}"
        print(f"template: {template}")

        texts_train = get_text_prompts_train(model, self.args, train_dataset, self.device, template=template)
        texts_list = get_text_prompts_val(val_dataset_list, val_dataset_name, template=template)

        return texts_train, texts_list

    def _run_evaluation(self, model, frozen_model, prompter, add_prompter,
                       val_loader_list, val_dataset_name, texts_list, texts_train):
        """Run evaluation on pre-trained model"""
        # Initialize wandb for evaluation
        init_wandb(
            project_name="CLIP_robust_finetune_text_enhanced",
            model_name=f'{self.args.baseline}-{self.args.template}-{self.args.attack}-{self.args.index}-ncaps{self.args.ncaps}',
            config=vars(self.args),
        )

        # Load baseline model
        model_setup = ModelSetup(self.args, self.device)
        model_setup.load_baseline_model(model)

        # Setup text prompts
        texts_list = get_text_prompts_val(val_dataset_list, val_dataset_name, template=self.args.template)
        print(f"template for evaluation: {self.args.template}")

        return validate(val_loader_list, val_dataset_name, texts_list, model, frozen_model,
                          self.device, prompter, add_prompter, self.args, self.args.epochs, None)


def train(train_loader, texts, model, frozen_model, prompter, add_prompter,
          optimizer, scheduler, scaler, epoch, device, args):
    """Train model for one epoch with adversarial training"""
    # Initialize metrics tracking
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    # Set models to training mode
    prompter.train()
    add_prompter.train()
    model.module.visual.train()

    # Training parameters
    num_batches_per_epoch = len(train_loader)
    alpha = args.train_stepsize
    attack_iters = args.train_numsteps
    end = time.time()

    # Prepare text tokens for training
    text_tokens = clip.tokenize(texts).to(device)
    text_tokens = text_tokens.view(args.ncaps, 200, -1)

    for i, (images, target, caption) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)

        # Update learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad()

        images, target = images.to(device), target.to(device)

        # Random target selection for diversity
        if args.random_target:
            torch.manual_seed(args.seed + step if args.seed is not None else 42)
            text_tokens = text_tokens[torch.randperm(args.ncaps)][0, :, :]

        with torch.autocast(device_type='cuda'):
            # Generate adversarial examples or use clean images
            if not args.VPbaseline:
                delta = pgd_CLIP(
                    prompter, model, add_prompter, images, target, text_tokens,
                    alpha, attack_iters, "l_inf", device=device, ncaps=args.ncaps,
                    epsilon=args.train_eps, seed=args.seed + step if args.seed is not None else None,
                )
                processed_images = clip_img_preprocessing(images + delta, device)
            else:
                processed_images = clip_img_preprocessing(images, device)

            # Enable gradient computation
            for param in model.parameters():
                param.requires_grad = True

            # Forward pass through prompters and model
            prompted_images = prompter(processed_images)
            clean_images = prompter(clip_img_preprocessing(images, device))
            prompt_token = add_prompter()

            logits_per_image, _, text_features = multiGPU_CLIP_classwise(
                model, prompted_images, text_tokens, prompt_token
            )
            text_features = text_features[:, target, :]

            # Calculate attention maps for loss computation
            attack_attention = attention_map_text(
                text_features, model, prompted_images, prompt_token, args
            ).view(prompted_images.size()[0], -1)

            clean_orig_attention = attention_map_text(
                text_features, frozen_model, clean_images, prompt_token, args
            ).view(prompted_images.size()[0], -1)

            clean_target_attention = attention_map_text(
                text_features, model, clean_images, prompt_token, args
            ).view(prompted_images.size()[0], -1)

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
