"""Model setup and initialization for SAFT training pipeline"""

import os
import copy
import torch
from models import clip, prompters
from models.prompters import TokenPrompter, NullPrompter
from models.model import multiGPU_CLIP_classwise
from helper import convert_models_to_fp32


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

    def load_baseline_model(self, model):
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
