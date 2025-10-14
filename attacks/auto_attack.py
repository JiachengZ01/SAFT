from models.model import *
from autoattack import AutoAttack
import functools

def autoattack_CLIP(model, images, target, text_tokens, device, attacks_to_run=['apgd-ce', 'apgd-dlr'], epsilon=0):
    """
    AutoAttack for CLIP model.
    """
    forward_pass = functools.partial(
        multiGPU_CLIP_image_logits,
        model=model, text_tokens=text_tokens, device=device
    )

    adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='standard', verbose=False, device=device)
    adversary.attacks_to_run = attacks_to_run
    x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    return x_adv