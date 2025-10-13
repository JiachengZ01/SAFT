import torch
from models.model import multiGPU_CLIP_classwise, clip_img_preprocessing, multiGPU_CLIP

lower_limit, upper_limit = 0, 1


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def pgd_CLIP(
    prompter,
    model,
    add_prompter,
    X,
    target,
    text_tokens,
    alpha,
    attack_iters,
    norm,
    device,
    ncaps,
    epsilon=2 / 255,
    seed=None,
):
    # Create generator for reproducible randomness if seed is provided
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    
    delta = torch.zeros_like(X).to(device)
    if norm == "l_inf":
        if generator is not None:
            delta = delta.uniform_(-epsilon, epsilon, generator=generator)
        else:
            delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        if generator is not None:
            delta = delta.normal_(generator=generator)
            r = torch.zeros_like(delta.view(delta.size(0), -1).norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)).uniform_(0, 1, generator=generator)
        else:
            delta.normal_()
            r = torch.zeros_like(delta.view(delta.size(0), -1).norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)).uniform_(0, 1)
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    target = target.repeat(ncaps)

    for _ in range(attack_iters):
        _images = clip_img_preprocessing(X + delta, device)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        prompt_token = add_prompter() if add_prompter is not None else None

        logits_per_image, _, _ = multiGPU_CLIP_classwise(model, prompted_images, text_tokens, prompt_token)  # (ncaps, bs, n_class)
        _, _, n_class = logits_per_image.shape
        logits_per_image = logits_per_image.view(-1, n_class)  # (ncaps * bs, n_class)
        CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction="mean").to(device)
        loss = CrossEntropyLoss(logits_per_image, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (
                (d + scaled_g * alpha)
                .view(d.size(0), -1)
                .renorm(p=2, dim=0, maxnorm=epsilon)
                .view_as(d)
            )
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()
    return delta
