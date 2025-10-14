"""Model setup and initialization for SAFT training pipeline"""

import os
import copy
import torch
from models import clip
from models.model import multiGPU_CLIP_classwise
from helper import convert_models_to_fp32

class ModelSetup:
    """Handles model initialization and optimizer setup"""

    def __init__(self, args, device):
        self.args = args
        self.device = device

    def create_models(self):
        """Create CLIP model and frozen reference"""
        print("Creating model...")

        if self.args.arch == "vit_b32":
            model, _ = clip.load("ViT-B/32", self.device, jit=False)
        elif self.args.arch == "vit_l14":
            model, _ = clip.load("ViT-L/14", self.device, jit=False)

        convert_models_to_fp32(model)
        model = model.to(self.device)
        frozen_model = copy.deepcopy(model).to(self.device)

        model = torch.nn.DataParallel(model)
        frozen_model = torch.nn.DataParallel(frozen_model)
        print(f"Using {torch.cuda.device_count()} GPUs for model parallelization.")

        model.eval()
        frozen_model.eval()

        return model, frozen_model

    def setup_optimizer(self, model):
        """Setup optimizer for fine-tuning"""
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

        return optimizer

    def load_checkpoint(self, model):
        """Load pre-trained model from checkpoint"""
        self.args.start_epoch = 0

        if self.args.resume and os.path.isfile(self.args.resume):
            print(f"=> loading checkpoint '{self.args.resume}'")
            loc = f"cuda:{self.args.gpu}" if self.args.gpu is not None else None
            checkpoint = torch.load(self.args.resume, map_location=loc)

            self.args.start_epoch = checkpoint["epoch"]

            if "vision_encoder_state_dict" in checkpoint.keys():
                model.module.visual.load_state_dict(
                    checkpoint["vision_encoder_state_dict"], strict=False
                )

            print(f"=> loaded checkpoint '{self.args.resume}' (epoch {checkpoint['epoch']})")
        elif self.args.resume:
            print(f"=> no checkpoint found at '{self.args.resume}'")

    def load_baseline_model(self, model):
        """Load baseline model for evaluation"""
        checkpoint_path = os.path.join(self.args.model_folder, f"model_final_seed{self.args.seed}.pth.tar")
        loc = f"cuda:{self.args.gpu}"
        checkpoint = torch.load(checkpoint_path, map_location=loc)
        model.module.visual.load_state_dict(checkpoint["vision_encoder_state_dict"], strict=False)

        print(f"checkpoint path: {checkpoint_path}")
