from __future__ import print_function
import numpy as np
import argparse
import os
import time
import random
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.datasets import *
from zmq import device
from models import prompters
from models.prompters import TokenPrompter, NullPrompter
from models.model import *
from models import clip
from attacks.pgd_attack import pgd_CLIP
from attacks.auto_attack import autoattack_CLIP
from attacks.cw_attack import attack_CW
import copy
from helper import *
import wandb


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

    config = Config()
    return config.parse_args()


class CLIPModel(torch.nn.Module):
    """CLIP model wrapper with visual prompting support"""
    def __init__(self, model, prompter, add_prompter, args):
        super().__init__()
        self.model = model
        self.prompter = prompter
        self.add_prompter = add_prompter
        self.args = args
    
    def forward(self, images, text_tokens, prompt_token=None):
        prompted_images = self.prompter(images)
        if prompt_token is None:
            prompt_token = self.add_prompter()
        return multiGPU_CLIP_classwise(self.model, prompted_images, text_tokens, prompt_token)


class ModelSetup:
    """Handles model initialization, prompters, and optimizer setup"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
    
    def create_models(self):
        """Create CLIP model and frozen reference"""
        add_prompt_len = getattr(self.args, 'add_prompt_size', 0) if self.args.adaptation_method == "VPT" else 0
        print("Creating model...")
        
        if self.args.arch == "vit_b32":
            model, _ = clip.load("ViT-B/32", self.device, jit=False, prompt_len=add_prompt_len)
        elif self.args.arch == "vit_l14":
            model, _ = clip.load("ViT-L/14", self.device, jit=False, prompt_len=add_prompt_len)
        
        convert_models_to_fp32(model)
        model = model.to(self.device)
        frozen_model = copy.deepcopy(model).to(self.device)
        
        model = torch.nn.DataParallel(model)
        frozen_model = torch.nn.DataParallel(frozen_model)
        print(f"Using {torch.cuda.device_count()} GPUs for model parallelization.")
        
        model.eval()
        frozen_model.eval()
        
        return model, frozen_model
    
    def setup_prompters_and_optimizer(self, model):
        """Setup visual prompters and optimizer based on adaptation method"""
        if self.args.adaptation_method == "VPT":
            prompter = prompters.__dict__[self.args.method](self.args).to(self.device)
            add_prompt_size = getattr(self.args, 'add_prompt_size', 10)
            add_prompter = TokenPrompter(add_prompt_size).to(self.device)
            params = list(prompter.parameters()) + list(add_prompter.parameters())
        else:
            prompter = NullPrompter().to(self.device)
            add_prompter = TokenPrompter(0).to(self.device)
            if self.args.last_num_ft == 0:
                params = model.module.visual.parameters()
            else:
                params = list(model.module.visual.parameters())[-self.args.last_num_ft:]
        
        optimizer = torch.optim.SGD(
            params,
            lr=self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        
        return prompter, add_prompter, optimizer
    
    def load_checkpoint(self, model, prompter, add_prompter):
        """Load pre-trained model from checkpoint"""
        self.args.start_epoch = 0
        best_acc_adv = 0
        best_acc_clean = 0
        
        if self.args.resume and os.path.isfile(self.args.resume):
            print(f"=> loading checkpoint '{self.args.resume}'")
            loc = f"cuda:{self.args.gpu}" if self.args.gpu is not None else None
            checkpoint = torch.load(self.args.resume, map_location=loc)
            
            self.args.start_epoch = checkpoint["epoch"]
            best_acc_adv = checkpoint.get("best_avg_acc_adv", 0)
            best_acc_clean = checkpoint.get("best_avg_acc_clean", 0)

            if "vision_encoder_state_dict" in checkpoint.keys():
                model.module.visual.load_state_dict(
                    checkpoint["vision_encoder_state_dict"], strict=False
                )
            else:
                prompter.load_state_dict(checkpoint["state_dict"])
                add_prompter.load_state_dict(checkpoint["add_prompter"])
            
            print(f"=> loaded checkpoint '{self.args.resume}' (epoch {checkpoint['epoch']})")
        elif self.args.resume:
            print(f"=> no checkpoint found at '{self.args.resume}'")
        
        return best_acc_adv, best_acc_clean


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
        
        base_datasets = ['tinyImageNet','cifar10', 'cifar100','STL10','Food101','oxfordpet',
                        'flowers102','dtd','EuroSAT','fgvc_aircraft','Caltech101','Caltech256',
                        'StanfordCars','PCAM']
        
        if self.args.autoattack:
            return base_datasets
        elif self.args.test_continue:
            return ['ImageNet','SUN397']
        else:
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
        self._load_baseline_model(model)
        
        # Setup text prompts
        texts_list = get_text_prompts_val(val_dataset_list, val_dataset_name, template=self.args.template)
        print(f"template for evaluation: {self.args.template}")
        
        # Run validation
        if self.args.text == "semantic_ensemble":
            self.args.ncaps = 5
            texts_train = get_text_prompts_train(model, self.args, val_loader_list[0].dataset, 
                                               self.device, template=self.args.template)
            return validate(val_loader_list, val_dataset_name, texts_list, model, frozen_model, 
                          self.device, prompter, add_prompter, self.args, self.args.epochs, texts_train)
        else:
            return validate(val_loader_list, val_dataset_name, texts_list, model, frozen_model, 
                          self.device, prompter, add_prompter, self.args, self.args.epochs, None)
    
    def _load_baseline_model(self, model):
        """Load baseline model for evaluation"""
        if self.args.baseline == "FARE":
            self.args.filename = "FARE"
            self.args.model_folder = os.path.join(self.args.model_dir, self.args.filename)
            checkpoint_path = os.path.join(self.args.model_folder, 
                                         "step_1500.pt" if self.args.arch == "vit_b32" 
                                         else "vitl14_fare_eps_1.pt")
            checkpoint = torch.load(checkpoint_path)
            model.module.visual.load_state_dict(checkpoint)
        else:
            if self.args.baseline == "PMG-AFT":
                self.args.filename = "PMG-AFT"
                self.args.model_folder = os.path.join(self.args.model_dir, self.args.filename)
            elif self.args.baseline == "TeCoA":
                self.args.model_folder = os.path.join(self.args.model_dir, self.args.filename)
            
            checkpoint_path = os.path.join(self.args.model_folder, f"model_best_seed{self.args.index}.pth.tar")
            loc = f"cuda:{self.args.gpu}"
            checkpoint = torch.load(checkpoint_path, map_location=loc)
            model.module.visual.load_state_dict(checkpoint["vision_encoder_state_dict"], strict=False)
        
        print(f"checkpoint path: {checkpoint_path}")


def main(args):
    """Main entry point for training"""
    global device
    device = torch.device(f"cuda:{args.gpu}")
    
    trainer = TrainingManager(args)
    trainer.run_training()

        # This section is now handled in TrainingManager._setup_data() and _run_training() 


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


class ValidationManager:
    """Handles model validation across multiple datasets with adversarial attacks"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
    
    def validate_all_datasets(self, val_loader_list, val_dataset_name, texts_list, 
                            model, frozen_model, prompter, add_prompter, epoch, text_train=None):
        """Run validation across all datasets"""
        adv_acc_list = []
        clean_acc_list = []
        
        for cnt, (val_loader, texts, dataset_name) in enumerate(zip(val_loader_list, texts_list, val_dataset_name)):
            adv_acc, clean_acc = self._validate_single_dataset(
                val_loader, texts, dataset_name, model, frozen_model,
                prompter, add_prompter, epoch, text_train
            )
            adv_acc_list.append(adv_acc)
            clean_acc_list.append(clean_acc)
        
        return np.mean(adv_acc_list), np.mean(clean_acc_list)
    
    def _validate_single_dataset(self, val_loader, texts, dataset_name, model, frozen_model,
                               prompter, add_prompter, epoch, text_train):
        """Validate on a single dataset"""
        # Setup attack configuration
        binary_datasets = ["PCAM", "hateful_memes"]
        attacks_to_run = ["apgd-ce"] if dataset_name in binary_datasets else ["apgd-ce", "apgd-dlr"]
        
        # Initialize metrics
        batch_time = AverageMeter("Time", ":6.3f")
        top1_clean = AverageMeter("Original Acc@1", ":6.2f")
        top1_adv = AverageMeter("Adv Original Acc@1", ":6.2f")
        
        # Set models to evaluation mode
        model.eval()
        frozen_model.eval()
        prompter.eval()
        add_prompter.eval()
        model.zero_grad()
        
        # Prepare text tokens
        text_tokens = self._prepare_text_tokens(texts, text_train)
        
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            images, target = images.to(self.device), target.to(self.device)
            
            with torch.autocast(device_type='cuda'):
                # Evaluate clean images
                clean_acc = self._evaluate_clean_images(model, images, target, text_tokens)
                top1_clean.update(clean_acc, images.size(0))
                
                # Generate and evaluate adversarial images
                attacked_images = self._generate_adversarial_images(
                    model, images, target, text_tokens, attacks_to_run, i
                )
                adv_acc = self._evaluate_adversarial_images(
                    model, attacked_images, target, text_tokens
                )
                top1_adv.update(adv_acc, images.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
        
        # Log results
        self._log_results(dataset_name, top1_clean.avg, top1_adv.avg, epoch)
        
        torch.cuda.empty_cache()
        print(f"{dataset_name} * Adversarial Acc@1 {top1_adv.avg:.3f} * Clean Acc@1 {top1_clean.avg:.3f}")
        
        return top1_adv.avg, top1_clean.avg
    
    def _prepare_text_tokens(self, texts, text_train):
        """Prepare text tokens based on text ensemble strategy"""
        if self.args.text == "semantic_ensemble" and text_train is not None:
            text_tokens = clip.tokenize(text_train).to(self.device)
            text_tokens = text_tokens.view(self.args.ncaps, 200, -1)
        else:
            text_tokens = clip.tokenize(texts).to(self.device)
            text_tokens = text_tokens.unsqueeze(0)
        return text_tokens
    
    def _evaluate_clean_images(self, model, images, target, text_tokens):
        """Evaluate model on clean images"""
        with torch.no_grad():
            output = multiGPU_CLIP_classwise(
                model, clip_img_preprocessing(images, self.device), text_tokens, None
            )[0]
            if self.args.text == "semantic_ensemble":
                output = output.mean(dim=0)
            acc = accuracy(output, target, topk=(1,))
            return acc[0].item()
    
    def _generate_adversarial_images(self, model, images, target, text_tokens, attacks_to_run, batch_idx):
        """Generate adversarial images using specified attack method"""
        if self.args.attack == "pgd":
            ncaps = self.args.ncaps if self.args.text == "semantic_ensemble" else 1
            delta = pgd_CLIP(
                None, model, None, images, target, text_tokens,
                self.args.test_stepsize, self.args.test_numsteps, "l_inf",
                self.device, ncaps, epsilon=self.args.test_eps,
                seed=self.args.seed + batch_idx if self.args.seed is not None else None,
            )
            return images + delta
        elif self.args.attack == "cw":
            cw_text_tokens = text_tokens.squeeze(0)
            delta = attack_CW(
                None, model, images, target, cw_text_tokens,
                self.args.test_stepsize, self.args.test_numsteps, "l_inf",
                self.device, epsilon=self.args.test_eps,
            )
            return images + delta
        else:  # autoattack
            auto_text_tokens = text_tokens.squeeze(0)
            return autoattack_CLIP(
                model, images, target, auto_text_tokens, self.device,
                attacks_to_run=attacks_to_run, epsilon=self.args.test_eps,
            )
    
    def _evaluate_adversarial_images(self, model, attacked_images, target, text_tokens):
        """Evaluate model on adversarial images"""
        with torch.no_grad():
            output_adv = multiGPU_CLIP_classwise(
                model, clip_img_preprocessing(attacked_images, self.device), text_tokens, None
            )[0]
            if self.args.text == "semantic_ensemble":
                output_adv = output_adv.mean(dim=0)
            acc = accuracy(output_adv, target, topk=(1,))
            return acc[0].item()
    
    def _log_results(self, dataset_name, clean_acc, adv_acc, epoch):
        """Log validation results to wandb"""
        if self.args.wandb:
            wandb.log({
                f"validation/{dataset_name}/test_clean_acc@1": clean_acc,
                f"validation/{dataset_name}/test_adversarial_acc@1": adv_acc,
                "epoch": epoch,
            })


def validate(val_loader_list, val_dataset_name, texts_list, model, frozen_model, 
            device, prompter, add_prompter, args, epoch, text_train=None):
    """Legacy wrapper for validation - delegates to ValidationManager"""
    validator = ValidationManager(args, device)
    return validator.validate_all_datasets(
        val_loader_list, val_dataset_name, texts_list, model, frozen_model,
        prompter, add_prompter, epoch, text_train
    )


if __name__ == "__main__":
    """Main entry point - parse args and run training"""
    args = parse_option()
    main(args)