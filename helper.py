import select
import shutil
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from val_datasets import (
    caltech, country211, dtd, eurosat, fgvc_aircraft, food101, flowers102,
    oxford_iiit_pet, pcam, stanford_cars, sun397,
)
import ssl
import random
import json
from itertools import chain
import sys
import wandb
from time import sleep
from PIL import Image
from models import clip

# SSL configuration for dataset downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Supported image extensions
IMG_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp",
)


def default_loader(path: str):
    """Default image loader for PIL images"""
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class UtilityManager:
    """Utility functions for model training and evaluation"""
    
    @staticmethod
    def init_wandb(project_name, model_name, config, **wandb_kwargs):
        """Initialize Weights & Biases with retry mechanism"""
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        while True:
            try:
                wandb_run = wandb.init(
                    project=project_name,
                    name=model_name,
                    save_code=True,
                    config=config,
                    **wandb_kwargs,
                )
                break
            except Exception as e:
                print("wandb connection error", file=sys.stderr)
                print(f"error: {e}", file=sys.stderr)
                sleep(1)
                print("retrying..", file=sys.stderr)
        return wandb_run

    @staticmethod
    def str2bool(v):
        """Convert string to boolean"""
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise ValueError

    @staticmethod
    def convert_models_to_fp32(model):
        """Convert model parameters to FP32 precision"""
        for p in model.parameters():
            p.data = p.data.float()
            if p.grad:
                p.grad.data = p.grad.data.float()

    @staticmethod
    def refine_classname(class_names):
        """Clean and standardize class names"""
        for i, class_name in enumerate(class_names):
            class_names[i] = (
                class_name.lower().replace("_", " ").replace("-", " ").replace("/", " ")
            )
        return class_names

    @staticmethod
    def save_checkpoint(state, args, is_final=False):
        """Save model checkpoint

        Args:
            state: Model state dict to save
            args: Configuration arguments
            is_final: If True, saves as final model; otherwise saves as temporary checkpoint
        """
        if is_final:
            checkpoint_path = os.path.join(
                args.model_folder, f"model_final_seed{args.seed}.pth.tar"
            )
            torch.save(state, checkpoint_path)
            print(f"Saved final model to {checkpoint_path}")
        else:
            checkpoint_path = os.path.join(
                args.model_folder, f"checkpoint_seed{args.seed}.pth.tar"
            )
            torch.save(state, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    @staticmethod
    def assign_learning_rate(optimizer, new_lr):
        """Update learning rate for optimizer"""
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    @staticmethod
    def _warmup_lr(base_lr, warmup_length, step):
        """Calculate warmup learning rate"""
        return base_lr * (step + 1) / warmup_length

    @classmethod
    def cosine_lr(cls, optimizer, base_lr, warmup_length, steps):
        """Create cosine learning rate scheduler with warmup"""
        def _lr_adjuster(step):
            if step < warmup_length:
                lr = cls._warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            cls.assign_learning_rate(optimizer, lr)
            return lr
        return _lr_adjuster

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Compute top-k accuracy"""
        output = output.squeeze(0)
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @staticmethod
    def load_imagenet_folder2name(path):
        """Load ImageNet folder to class name mapping"""
        dict_imagenet_folder2name = {}
        with open(path) as f:
            for line in f:
                split_name = line.strip().split()
                if len(split_name) >= 3:
                    cat_name = split_name[2]
                    folder_id = split_name[0]
                    dict_imagenet_folder2name[folder_id] = cat_name
        return dict_imagenet_folder2name

    @staticmethod
    def one_hot_embedding(labels, num_classes, device):
        """Convert labels to one-hot encoding"""
        y = torch.eye(num_classes).to(device)
        return y[labels]


class AverageMeter:
    """Compute and store the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all values"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update with new value"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


# Legacy function wrappers for backward compatibility
init_wandb = UtilityManager.init_wandb
str2bool = UtilityManager.str2bool
convert_models_to_fp32 = UtilityManager.convert_models_to_fp32
refine_classname = UtilityManager.refine_classname
save_checkpoint = UtilityManager.save_checkpoint
assign_learning_rate = UtilityManager.assign_learning_rate
cosine_lr = UtilityManager.cosine_lr
accuracy = UtilityManager.accuracy
load_imagenet_folder2name = UtilityManager.load_imagenet_folder2name
one_hot_embedding = UtilityManager.one_hot_embedding


class DatasetManager:
    """Handles dataset loading and preprocessing for training and validation"""
    
    def __init__(self):
        # Standard preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()
        ])
        
        self.preprocess224 = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor()
        ])
        
        # Dataset configuration mapping
        self.dataset_configs = {
            "cifar100": {"class": CIFAR100, "path": "./datasets/CIFAR100", "transform": "preprocess"},
            "cifar10": {"class": CIFAR10, "path": "./datasets/CIFAR10", "transform": "preprocess"},
            "ImageNet": {"class": ImageFolder, "path": "/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/train", "transform": "preprocess224"},
            "tinyImageNet": {"class": ImageFolder, "path": "./data/tiny-imagenet-200/train/", "transform": "preprocess224"}
        }
    
    def load_train_dataset(self, args):
        """Load training dataset based on configuration"""
        dataset_name = args.dataset
        
        if dataset_name not in self.dataset_configs:
            print(f"Train dataset {dataset_name} not implemented")
            raise NotImplementedError
        
        config = self.dataset_configs[dataset_name]
        transform = getattr(self, config["transform"])
        
        if dataset_name in ["cifar100", "cifar10"]:
            print(f"Loading {dataset_name} from {config['path']}")
            return config["class"](
                config["path"], transform=transform, download=True, train=True
            )
        elif dataset_name == "ImageNet":
            print(f"Loading ImageNet from {config['path']}")
            return config["class"](config["path"], transform=transform)
        elif dataset_name == "tinyImageNet":
            print(f"Loading tinyImageNet from {config['path']}")
            return config["class"](config["path"], transform=transform)
    
    def load_val_datasets(self, args, val_dataset_names):
        """Load multiple validation datasets"""
        val_dataset_list = []
        
        for dataset_name in val_dataset_names:
            dataset = self._load_single_val_dataset(dataset_name)
            val_dataset_list.append(dataset)
        
        return val_dataset_list
    
    def _load_single_val_dataset(self, dataset_name):
        """Load a single validation dataset"""
        dataset_loaders = {
            "cifar10": lambda: CIFAR10("./datasets/CIFAR10", transform=self.preprocess, download=True, train=False),
            "cifar100": lambda: CIFAR100("./datasets/CIFAR100", transform=self.preprocess, download=True, train=False),
            "Caltech101": lambda: caltech.Caltech101("./datasets/", target_type="category", transform=self.preprocess224, download=True),
            "PCAM": lambda: pcam.PCAM("./datasets/", split="test", transform=self.preprocess224, download=False),
            "STL10": lambda: STL10("./datasets/STL10", split="test", transform=self.preprocess, download=True),
            "SUN397": lambda: sun397.SUN397("./datasets/", transform=self.preprocess224, download=True),
            "StanfordCars": lambda: stanford_cars.StanfordCars("./datasets/", split="test", transform=self.preprocess224, download=False),
            "Food101": lambda: food101.Food101("./datasets/", split="test", transform=self.preprocess224, download=True),
            "oxfordpet": lambda: oxford_iiit_pet.OxfordIIITPet("./datasets/", split="test", transform=self.preprocess224, download=True),
            "EuroSAT": lambda: eurosat.EuroSAT("./datasets/", transform=self.preprocess224, download=True),
            "Caltech256": lambda: caltech.Caltech256("./datasets/", transform=self.preprocess224, download=True),
            "flowers102": lambda: flowers102.Flowers102("./datasets/", split="test", transform=self.preprocess224, download=True),
            "Country211": lambda: country211.Country211("./datasets/", split="test", transform=self.preprocess224, download=True),
            "dtd": lambda: dtd.DTD("./datasets/", split="test", transform=self.preprocess224, download=True),
            "fgvc_aircraft": lambda: fgvc_aircraft.FGVCAircraft("./datasets/", split="test", transform=self.preprocess224, download=True),
            "ImageNet": lambda: ImageFolder("./data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/val", transform=self.preprocess224),
            "tinyImageNet": lambda: ImageFolder("./data/tiny-imagenet-200/val/", transform=self.preprocess224),
        }
        
        if dataset_name not in dataset_loaders:
            print(f"Val dataset {dataset_name} not implemented")
            raise NotImplementedError
        
        return dataset_loaders[dataset_name]()


# Legacy function wrappers for backward compatibility
def load_train_dataset(args):
    """Legacy wrapper for loading training dataset"""
    dataset_manager = DatasetManager()
    return dataset_manager.load_train_dataset(args)

def load_val_datasets(args, val_dataset_names):
    """Legacy wrapper for loading validation datasets"""
    dataset_manager = DatasetManager()
    return dataset_manager.load_val_datasets(args, val_dataset_names)

# Keep the original preprocessing transforms as module-level variables for compatibility
preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
preprocess224 = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
)


# This section is now handled by DatasetManager class


class TextPromptManager:
    """Manages text prompt generation for training and validation"""
    
    @staticmethod
    def get_text_prompts_val(val_dataset_list, val_dataset_name, template="This is a photo of a {}"):
        """Generate text prompts for validation datasets"""
        texts_list = []
        
        for cnt, dataset in enumerate(val_dataset_list):
            dataset_name = val_dataset_name[cnt]
            
            # Use pre-defined prompts if available
            if hasattr(dataset, "clip_prompts"):
                texts_tmp = dataset.clip_prompts
            else:
                # Extract class names from dataset
                try:
                    class_names = dataset.classes
                except AttributeError:
                    class_names = dataset.categories
                
                # Convert folder names to readable class names for ImageNet variants
                class_names = TextPromptManager._convert_class_names(class_names, dataset_name)
                class_names = UtilityManager.refine_classname(class_names)
                
                # Generate prompts using template
                texts_tmp = [template.format(label) for label in class_names]
            
            texts_list.append(texts_tmp)
        
        assert len(texts_list) == len(val_dataset_list)
        return texts_list
    
    @staticmethod
    def _convert_class_names(class_names, dataset_name):
        """Convert dataset-specific class identifiers to readable names"""
        imagenet_variants = ["ImageNet", "ImageNet-A", "ImageNet-R", "ImageNet-O"]
        
        if dataset_name in imagenet_variants:
            folder2name = UtilityManager.load_imagenet_folder2name(
                "./utils/imagenet_classes_names.txt"
            )
            return [folder2name.get(class_name, class_name) for class_name in class_names]
        elif dataset_name == "tinyImageNet":
            folder2name = UtilityManager.load_imagenet_folder2name(
                "./utils/tinyimagenet_classes_name.txt"
            )
            return [folder2name.get(class_name, class_name) for class_name in class_names]
        
        return class_names
    
    @staticmethod
    def select_top_n_descriptions(model, class_name, descriptions, n, device):
        """Select top-n most relevant descriptions using CLIP similarity"""
        text_tokens = clip.tokenize(descriptions).to(device)
        class_tokens = clip.tokenize([class_name]).to(device)

        # Encode text features
        text_features = model.module.encode_text(text_tokens)
        class_features = model.module.encode_text(class_tokens)

        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = text_features @ class_features.T

        # Select top-n descriptions
        _, top_n_idx = torch.topk(similarity.squeeze(-1), n, dim=0)
        return [descriptions[idx] for idx in top_n_idx]
    
    @staticmethod
    def get_text_prompts_train(model, args, train_dataset, device, template="This is a photo of a {}"):
        """Generate text prompts for training dataset with multiple caption support"""
        class_names = train_dataset.classes
        
        # Convert class identifiers to readable names
        class_names = TextPromptManager._convert_class_names(class_names, args.dataset)
        class_names = UtilityManager.refine_classname(class_names)

        # Simple template-based prompts for baseline methods
        if args.ncaps == 0:
            args.ncaps = 1  # Ensure dimension compatibility
            return [template.format(class_name) for class_name in class_names]
        
        # Enhanced prompts with multiple descriptions
        return TextPromptManager._generate_enhanced_prompts(model, args, class_names, device)
    
    @staticmethod
    def _generate_enhanced_prompts(model, args, class_names, device):
        """Generate enhanced prompts with multiple descriptions per class"""
        # Load description database
        if args.MLLM:
            if args.dataset == "tinyImageNet":
                description_file = "prompts/tinyimagenet_descriptions_by_category.json"
            elif args.dataset == "ImageNet":
                description_file = "prompts/imagenet_descriptions_by_category.json"
            else:
                description_file = "prompts/cupl.json"  # fallback
        else:
            description_file = "prompts/cupl.json"
        
        try:
            with open(description_file, "r") as f:
                descriptions = json.load(f)
        except FileNotFoundError:
            print(f"Description file {description_file} not found, using template prompts")
            return [[f"This is a photo of a {class_name}"] * args.ncaps for class_name in class_names]

        texts_train = []
        for class_name in class_names:
            matched_class = TextPromptManager._find_best_match(class_name, descriptions)
            
            if matched_class and len(descriptions[matched_class]) >= args.ncaps:
                # Select top-n most relevant descriptions
                selected_descriptions = TextPromptManager.select_top_n_descriptions(
                    model, matched_class, descriptions[matched_class], args.ncaps, device
                )
                texts_train.append(selected_descriptions)
            else:
                # Fallback to template-based prompts
                texts_train.append([f"This is a photo of a {class_name}"] * args.ncaps)

        # Flatten according to columns (transpose and flatten)
        return list(chain.from_iterable(zip(*texts_train)))
    
    @staticmethod
    def _find_best_match(class_name, descriptions):
        """Find best matching class name in description database"""
        if class_name in descriptions:
            return class_name

        # Fuzzy matching
        for key in descriptions.keys():
            if class_name in key or key in class_name:
                return key
        return None
    
    @staticmethod
    def convert_keys_to_lowercase(d):
        """Convert dictionary keys to lowercase"""
        return {key.lower(): value for key, value in d.items()}


# Legacy function wrappers for backward compatibility
get_text_prompts_val = TextPromptManager.get_text_prompts_val
get_text_prompts_train = TextPromptManager.get_text_prompts_train
select_top_n_descriptions = TextPromptManager.select_top_n_descriptions
find_best_match = TextPromptManager._find_best_match
convert_keys_to_lowercase = TextPromptManager.convert_keys_to_lowercase


class AttentionMapProcessor:
    """Processes attention maps for text-guided adversarial training"""
    
    @staticmethod
    def compute_attention_map(text_features, clip_model, images, args):
        """Compute text-guided attention map from image and text features"""
        # Extract and normalize image features
        image_features = clip_model.module.encode_image(images, None)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        # Get spatial features (excluding CLS token)
        img_spatial_feat = image_features[:, 1:, :]  # Shape: (bs, 49, 512)
        
        # Expand spatial features for multiple captions
        img_spatial_feat = img_spatial_feat.unsqueeze(0).expand(
            text_features.size(0), -1, -1, -1
        )  # Shape: (ncaps, bs, 49, 512)

        # Compute attention through matrix multiplication
        # (ncaps, bs, 49, 512) @ (ncaps, bs, 512, 1) -> (ncaps, bs, 49, 1)
        attention_map = torch.einsum(
            "bijk, bikl -> bijl", img_spatial_feat, text_features.unsqueeze(-1)
        )

        # Average across multiple captions
        # (ncaps, bs, 49, 1) -> (bs, 49, 1)
        attention_map = attention_map.mean(dim=0)

        # Normalize attention values to [0, 1] range
        attention_map = AttentionMapProcessor._normalize_attention(attention_map)

        # Reshape and interpolate to image size
        return AttentionMapProcessor._reshape_and_interpolate(attention_map, args.image_size)
    
    @staticmethod
    def _normalize_attention(attention_map):
        """Normalize attention values to [0, 1] range"""
        min_vals = attention_map.min(1, keepdim=True)[0]
        max_vals = attention_map.max(1, keepdim=True)[0]
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
        
        return (attention_map - min_vals) / range_vals
    
    @staticmethod
    def _reshape_and_interpolate(attention_map, target_size):
        """Reshape attention map from flattened spatial to 2D and interpolate to target size"""
        # Calculate spatial dimensions (assuming square feature maps)
        spatial_dim = int(attention_map.shape[1] ** 0.5)
        
        # Reshape: (bs, 49, 1) -> (bs, 1, 7, 7)
        attention_map = attention_map.reshape(
            attention_map.shape[0], spatial_dim, spatial_dim, -1
        ).permute(0, 3, 1, 2)

        # Interpolate to target image size: (bs, 1, 7, 7) -> (bs, 1, target_size, target_size)
        return torch.nn.functional.interpolate(
            attention_map, target_size, mode="bilinear", align_corners=False
        )


# Legacy function wrapper for backward compatibility
def attention_map_text(text_features, clip_model, images, args):
    """Legacy wrapper for attention map computation"""
    return AttentionMapProcessor.compute_attention_map(
        text_features, clip_model, images, args
    )


class LossManager:
    """Manages multi-component loss computation for adversarial training"""
    
    def __init__(self, device):
        self.device = device
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean").to(device)
        self.l1_loss = torch.nn.L1Loss(reduction="mean")
    
    def compute_total_loss(self, logits_per_image, target, adv_attention, 
                          clean_attention, clean_attention_model, args):
        """Compute the total loss with classification and attention map components"""
        # Classification loss
        loss_classification = self._compute_classification_loss(logits_per_image, target)
        
        # Attention map losses
        loss_adversarial_reg = self._compute_attention_loss(
            adv_attention, clean_attention, args.Distance_metric
        )
        loss_model_consistency = self._compute_attention_loss(
            clean_attention_model, clean_attention, args.Distance_metric
        )
        
        # Weighted combination
        weighted_adv_reg = args.Alpha * loss_adversarial_reg
        weighted_model_consistency = args.Beta * loss_model_consistency
        
        return loss_classification, weighted_adv_reg, weighted_model_consistency
    
    def _compute_classification_loss(self, logits_per_image, target):
        """Compute cross-entropy classification loss"""
        ncaps, batch_size, n_class = logits_per_image.shape
        
        # Reshape logits and targets for loss computation
        output = logits_per_image.view(-1, n_class)  # (ncaps * bs, n_class)
        target_repeated = target.repeat(ncaps)  # (ncaps * bs,)
        
        return self.cross_entropy_loss(output, target_repeated)
    
    def _compute_attention_loss(self, attention1, attention2, distance_metric):
        """Compute attention map distance loss using specified metric"""
        if distance_metric == "cos":
            return self._cosine_distance_loss(attention1, attention2)
        elif distance_metric == "l2":
            return self._l2_distance_loss(attention1, attention2)
        elif distance_metric == "l1":
            return self._l1_distance_loss(attention1, attention2)
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
    
    def _cosine_distance_loss(self, attention1, attention2):
        """Compute cosine distance loss between attention maps"""
        # Flatten attention maps for cosine similarity computation
        attention1_flat = attention1.view(attention1.size(0), -1)
        attention2_flat = attention2.view(attention2.size(0), -1)
        
        cosine_sim = torch.nn.functional.cosine_similarity(
            attention1_flat, attention2_flat, dim=1, eps=1e-8
        )
        return torch.mean(1 - cosine_sim)
    
    def _l2_distance_loss(self, attention1, attention2):
        """Compute L2 distance loss between attention maps"""
        attention1_flat = attention1.view(attention1.size(0), -1)
        attention2_flat = attention2.view(attention2.size(0), -1)
        return torch.mean(torch.norm(attention1_flat - attention2_flat, dim=1, p=2))
    
    def _l1_distance_loss(self, attention1, attention2):
        """Compute L1 distance loss between attention maps"""
        return self.l1_loss(attention1, attention2)


# Legacy function wrapper for backward compatibility
def criterion(logits_per_image, target, adv_atten, clean_atten, clean_atten_model, device, args):
    """Legacy wrapper for loss computation"""
    loss_manager = LossManager(device)
    return loss_manager.compute_total_loss(
        logits_per_image, target, adv_atten, clean_atten, clean_atten_model, args
    )