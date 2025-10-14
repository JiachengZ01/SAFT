"""Validation logic for SAFT training pipeline"""

import torch
import numpy as np
from tqdm import tqdm
from models import clip
from models.model import multiGPU_CLIP_classwise, clip_img_preprocessing
from attacks.pgd_attack import pgd_CLIP
from attacks.auto_attack import autoattack_CLIP
from attacks.cw_attack import attack_CW
from helper import AverageMeter, accuracy
import wandb


class ValidationManager:
    """Handles model validation across multiple datasets with adversarial attacks"""

    def __init__(self, args, device):
        self.args = args
        self.device = device

    def validate_all_datasets(self, val_loader_list, val_dataset_name, texts_list,
                            model, frozen_model, epoch, text_train=None):
        """Run validation across all datasets"""
        adv_acc_list = []
        clean_acc_list = []

        for cnt, (val_loader, texts, dataset_name) in enumerate(zip(val_loader_list, texts_list, val_dataset_name)):
            adv_acc, clean_acc = self._validate_single_dataset(
                val_loader, texts, dataset_name, model, frozen_model,
                epoch, text_train
            )
            adv_acc_list.append(adv_acc)
            clean_acc_list.append(clean_acc)

        return np.mean(adv_acc_list), np.mean(clean_acc_list)

    def _validate_single_dataset(self, val_loader, texts, dataset_name, model, frozen_model,
                               epoch, text_train):
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
        model.zero_grad()

        # Prepare text tokens
        text_tokens = self._prepare_text_tokens(texts, text_train)

        import time
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            images, target = images.to(self.device), target.to(self.device)

            # Evaluate clean images with autocast
            with torch.autocast(device_type='cuda'):
                clean_acc = self._evaluate_clean_images(model, images, target, text_tokens)
                top1_clean.update(clean_acc, images.size(0))

            # Generate adversarial images WITHOUT autocast (pgd needs gradients)
            attacked_images = self._generate_adversarial_images(
                model, images, target, text_tokens, attacks_to_run, i
            )

            # Evaluate adversarial images with autocast
            with torch.autocast(device_type='cuda'):
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
                model, clip_img_preprocessing(images, self.device), text_tokens
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
                model, images, target, text_tokens,
                self.args.test_stepsize, self.args.test_numsteps, "l_inf",
                self.device, ncaps, epsilon=self.args.test_eps,
            )
            return images + delta
        elif self.args.attack == "cw":
            cw_text_tokens = text_tokens.squeeze(0)
            delta = attack_CW(
                model, images, target, cw_text_tokens,
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
                model, clip_img_preprocessing(attacked_images, self.device), text_tokens
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
            device, args, epoch, text_train=None):
    """Legacy wrapper for validation - delegates to ValidationManager"""
    validator = ValidationManager(args, device)
    return validator.validate_all_datasets(
        val_loader_list, val_dataset_name, texts_list, model, frozen_model,
        epoch, text_train
    )
